import asyncio
import os
import sys
import time
from typing import Dict, Optional, List, Any
import uuid

import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.google import GoogleLLMService, BreezeflowLLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from deepgram import LiveOptions
from pipecat.vad.vad_analyzer import VADParams


load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Store active agent instances
active_agents: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="Agent API")


class AgentRequest(BaseModel):
    """Request model for starting a new agent instance with specified configuration."""
    agent_id: str
    room_url: Optional[str] = None
    system_prompt: Optional[str] = "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way."
    voice: Optional[str] = "aura-helios-en"


class AgentResponse(BaseModel):
    agent_id: str
    instance_id: str
    status: str
    room_url: str  # Make room_url required instead of optional
    room_name: Optional[str] = None
    created_at: Optional[str] = None
    room_config: Optional[Dict[str, Any]] = None


async def create_agent_instance(agent_id: str, agent_config: AgentRequest) -> Dict[str, Any]:
    """Create and start an agent instance."""
    instance_id = str(uuid.uuid4())

    # Create a session for this agent instance
    session = aiohttp.ClientSession()

    # Configure room URL or create a new one
    room_url = agent_config.room_url
    if not room_url:
        room_url, _ = await configure(session)

    # Create transport
    transport = DailyTransport(
        room_url,
        None,
        f"Agent {agent_id}",
        DailyParams(
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=1)),
            vad_audio_passthrough=True,
        ),
    )

    # Create STT service
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            model="nova-2-general",
            language="en-US",
            smart_format=True,
            vad_events=True
        )
    )

    # Create TTS service
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice=agent_config.voice,
        sample_rate=24000
    )

    # Create LLM service based on agent_id
    llm = BreezeflowLLMService(
        params=BreezeflowLLMService.InputParams(
            chatbot_id=agent_id
        )
    )

    # Initialize conversation context
    messages = [
        {
            "role": "system",
            "content": agent_config.system_prompt,
        },
    ]

    # Create context and aggregator
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # Set up pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    # Create pipeline task
    task = PipelineTask(
        pipeline,
        PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
        ),
    )

    # Set up event handlers
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        await transport.capture_participant_transcription(participant["id"])
        # Kick off the conversation
        messages.append(
            {"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"Participant left: {participant['id']}")
        if len(transport.participants) <= 1:  # Only the bot remains
            logger.info(
                f"Stopping agent instance {instance_id} as all participants left")
            await stop_agent(instance_id)

    # Create runner
    runner = PipelineRunner()

    # Start the pipeline in the background
    asyncio.create_task(runner.run(task))

    # Store all components for later reference
    agent_instance = {
        "instance_id": instance_id,
        "agent_id": agent_id,
        "session": session,
        "transport": transport,
        "task": task,
        "runner": runner,
        "room_url": room_url,
        "context": context,
        "messages": messages,
        "status": "running"
    }

    return agent_instance


async def stop_agent(instance_id: str) -> None:
    """Stop an agent instance and clean up resources."""
    if instance_id not in active_agents:
        return

    agent = active_agents[instance_id]

    try:
        # Cancel the pipeline task
        if agent["task"]:
            await agent["task"].cancel()

        # Close the session
        if agent["session"]:
            await agent["session"].close()

        # Update status
        agent["status"] = "stopped"

        # Remove from active agents
        del active_agents[instance_id]

        logger.info(f"Agent instance {instance_id} stopped and cleaned up")
    except Exception as e:
        logger.error(f"Error stopping agent instance {instance_id}: {e}")


@app.get("/")
async def read_root():
    """Return a welcome message for the Agent API root endpoint."""
    return {"message": "Welcome to the Agent API"}


async def create_daily_room() -> Dict[str, Any]:
    """Create a new Daily.co room with 30-minute expiration."""
    daily_api_key = os.getenv("DAILY_API_KEY")
    if not daily_api_key:
        raise HTTPException(status_code=500, detail="DAILY_API_KEY not configured")

    url = "https://api.daily.co/v1/rooms/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {daily_api_key}"
    }
    data = {
        "properties": {
            "exp": int(time.time()) + 1800  # 30 minutes in seconds
        }
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            if not response.ok:
                error_text = await response.text()
                logger.error(f"Failed to create Daily room: {error_text}")
                raise HTTPException(status_code=response.status,
                                    detail="Failed to create Daily room")

            return await response.json()


@app.post("/agents/start", response_model=AgentResponse)
async def start_agent(agent_request: AgentRequest, background_tasks: BackgroundTasks):
    """Start a new agent instance."""
    try:
        # Create a room if one wasn't provided
        room_data = None
        room_url = agent_request.room_url
        if not room_url:
            room_data = await create_daily_room()
            room_url = room_data["url"]

        # Create the agent instance
        agent_instance = await create_agent_instance(agent_request.agent_id, agent_request)
        instance_id = agent_instance["instance_id"]

        # Store in active agents
        active_agents[instance_id] = agent_instance

        logger.info(
            f"Started agent instance {instance_id} for agent {agent_request.agent_id}")

        # Return response with room details if available
        response = AgentResponse(
            agent_id=agent_request.agent_id,
            instance_id=instance_id,
            status="running",
            room_url=room_url
        )

        # Add additional room details if we created the room
        if room_data:
            response.room_name = room_data.get("name")
            response.created_at = room_data.get("created_at")
            response.room_config = room_data.get("config")

        return response
    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start agent: {str(e)}")


def main():
    """Run the FastAPI server."""
    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

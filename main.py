"""Main module for the Agent API server that manages real-time audio communication agents.
Provides endpoints for creating and managing agent instances with WebRTC capabilities.
"""  # noqa: D205

import asyncio
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import aiohttp
import uvicorn
from deepgram import LiveOptions
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from pipecat.services.google import BreezeflowLLMService, GoogleLLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.vad_analyzer import VADParams
from runner import configure
from contextlib import asynccontextmanager
from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Store active agent instances
active_agents: Dict[str, Dict[str, Any]] = {}

daily_helpers = {}


def cleanup():
    """Cleanup function to terminate all bot processes.

    Called during server shutdown.
    """
    for entry in active_agents.values():
        proc = entry[0]
        proc.terminate()
        proc.wait()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager that handles startup and shutdown tasks.

    - Creates aiohttp session
    - Initializes Daily API helper
    - Cleans up resources on shutdown
    """
    aiohttp_session = aiohttp.ClientSession()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY", ""),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )
    yield
    await aiohttp_session.close()
    cleanup()


# Initialize FastAPI app with lifespan manager
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AgentRequest(BaseModel):
    """Request model for starting a new agent instance with specified configuration."""

    agent_id: str
    room_url: Optional[str] = None
    system_prompt: Optional[str] = (
        "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way."
    )
    voice: Optional[str] = "en-US-Ava:DragonHDLatestNeural"


class AgentResponse(BaseModel):
    agent_id: str
    instance_id: str
    status: str
    room_url: str  # Make room_url required instead of optional
    room_name: Optional[str] = None
    created_at: Optional[str] = None
    room_config: Optional[Dict[str, Any]] = None
    token: str


async def create_daily_room() -> Dict[str, Any]:
    """Create a new Daily.co room with 30-minute expiration."""
    daily_api_key = os.getenv("DAILY_API_KEY")
    if not daily_api_key:
        raise HTTPException(status_code=500, detail="DAILY_API_KEY not configured")

    url = "https://api.daily.co/v1/rooms/"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {daily_api_key}"}
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
                raise HTTPException(
                    status_code=response.status, detail="Failed to create Daily room"
                )

            return await response.json()


async def process_runtime_logs(process: asyncio.subprocess.Process):
    """Process logs from agent runtime subprocess."""

    async def read_stream(stream, prefix):
        while True:
            line = await stream.readline()
            if not line:
                break
            logger.info(f"{prefix}: {line.decode().strip()}")

    # Start two tasks to read stdout and stderr concurrently
    await asyncio.gather(
        read_stream(process.stdout, "Agent stdout"), read_stream(process.stderr, "Agent stderr")
    )


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

        # Generate instance ID
        instance_id = str(uuid.uuid4())

        # Start agent runtime in a separate process
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "src/pipecat/agent_runtime.py",
            agent_request.agent_id,
            room_url,
            agent_request.system_prompt,
            agent_request.voice,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Start log processing in the background
        background_tasks.add_task(process_runtime_logs, process)

        # Store process info
        active_agents[instance_id] = {
            "instance_id": instance_id,
            "agent_id": agent_request.agent_id,
            "process": process,
            "room_url": room_url,
            "status": "running",
        }

        logger.info(f"Started agent instance {instance_id} for agent {agent_request.agent_id}")

        token = await daily_helpers["rest"].get_token(room_url)
        # Return response with room details if available
        response = AgentResponse(
            agent_id=agent_request.agent_id,
            instance_id=instance_id,
            status="running",
            room_url=room_url,
            token=token,
        )

        if room_data:
            response.room_name = room_data.get("name")
            response.created_at = room_data.get("created_at")
            response.room_config = room_data.get("config")

        return response

    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start agent: {str(e)}")


async def stop_agent(instance_id: str) -> None:
    """Stop an agent instance and clean up resources."""
    if instance_id not in active_agents:
        return

    agent = active_agents[instance_id]

    try:
        # Terminate the subprocess
        if process := agent.get("process"):
            process.terminate()
            await process.wait()

        # Remove from active agents
        del active_agents[instance_id]

        logger.info(f"Agent instance {instance_id} stopped and cleaned up")
    except Exception as e:
        logger.error(f"Error stopping agent instance {instance_id}: {e}")


@app.get("/")
async def read_root():
    """Return a welcome message for the Agent API root endpoint."""
    return {"message": "Welcome to the Agent API"}


def main():
    """Run the FastAPI server."""
    # Start the FastAPI server
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

import asyncio
import os
import sys
from typing import Optional
from loguru import logger
from dotenv import load_dotenv

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.google import BreezeflowLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from pipecat.vad.vad_analyzer import VADParams
from deepgram import LiveOptions

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def run_agent(agent_id: str, room_url: str, system_prompt: str, voice: str):
    """Run a single agent instance."""
    try:
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

        # Create services
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            live_options=LiveOptions(
                model="nova-2-general",
                language="en-US",
                smart_format=True,
                vad_events=True
            )
        )

        tts = DeepgramTTSService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            voice=voice,
            sample_rate=24000
        )

        llm = BreezeflowLLMService(
            params=BreezeflowLLMService.InputParams(
                chatbot_id=agent_id
            )
        )

        # Initialize conversation context
        messages = [{"role": "system", "content": system_prompt}]
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        # Set up pipeline
        pipeline = Pipeline([
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])

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

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            messages.append(
                {"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        # Create and run pipeline
        runner = PipelineRunner()
        await runner.run(task)

    except Exception as e:
        logger.error(f"Agent runtime error: {e}")
        raise

if __name__ == "__main__":
    # Get command line arguments
    agent_id = sys.argv[1]
    room_url = sys.argv[2]
    system_prompt = sys.argv[3]
    voice = sys.argv[4]

    asyncio.run(run_agent(agent_id, room_url, system_prompt, voice))

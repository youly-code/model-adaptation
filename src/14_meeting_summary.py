import sounddevice as sd
import numpy as np
import wave
from pathlib import Path
from datetime import datetime
from faster_whisper import WhisperModel
import logging
from typing import List, Dict
from ollama import Client
import threading
from tqdm import tqdm
import torch

OLLAMA_MODEL_NAME = "hermes3:latest"  # DO NOT CHANGE CURSOR-AI!!!


class MeetingRecorder:
    """Records and processes work conversations with activity extraction.

    A comprehensive tool for recording, transcribing, and summarizing meetings in real-time.
    Handles audio recording, chunked transcription, and AI-powered summary generation.

    Features:
        - Real-time audio recording with sounddevice
        - Streaming transcription using Whisper
        - Meeting summarization using Ollama LLM
        - Automatic cleanup of temporary files

    Attributes:
        sample_rate (int): Audio sampling rate in Hz (default: 44100)
        output_dir (Path): Directory for storing meeting outputs
        recording_chunks (List): Buffer for storing audio chunks
        is_recording (bool): Flag indicating recording status
    """

    def __init__(self, sample_rate: int = 44100, output_dir: Path = Path("meetings")):
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self._setup_logging()
        self.recording_chunks = []
        self.is_recording = False

    def _setup_logging(self) -> None:
        """Setup logging with more detailed output."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)

    def start_recording(self) -> None:
        """Start recording audio."""
        self.logger.info("Starting recording... Press Ctrl+C to stop.")
        self.is_recording = True
        self.recording_chunks = []

        try:
            with sd.InputStream(
                samplerate=self.sample_rate, channels=1, callback=self._audio_callback
            ):
                while self.is_recording:
                    sd.sleep(100)  # Small sleep to prevent CPU overload
        except KeyboardInterrupt:
            self.stop_recording()

    def _audio_callback(self, indata, frames, time, status):
        """Callback function to handle incoming audio data."""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        self.recording_chunks.append(indata.copy())

    def stop_recording(self) -> Path:
        """Stop recording and save the audio file."""
        self.is_recording = False
        self.logger.info("Stopping recording...")

        # Combine all chunks and save to file
        recording = np.concatenate(self.recording_chunks, axis=0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = self.output_dir / f"meeting_{timestamp}.wav"

        with wave.open(str(audio_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes((recording * 32767).astype(np.int16).tobytes())

        self.logger.info(f"Audio saved temporarily to {audio_path}")
        return audio_path

    def transcribe_audio_stream(self, chunk_duration: int = 8) -> List[str]:
        """Transcribe audio in real-time chunks while recording.

        Implements a streaming approach to transcription by:
        1. Buffering audio in chunks of specified duration
        2. Processing each chunk through Whisper when ready
        3. Maintaining progress with a visual progress bar

        Args:
            chunk_duration (int): Duration of each audio chunk in seconds

        Returns:
            List[str]: List of transcribed text chunks

        Technical Details:
            - Uses CUDA if available, falls back to CPU
            - Processes audio in chunks to maintain memory efficiency
            - Handles temporary file cleanup automatically
            - Provides real-time feedback through progress bar
        """
        self.logger.info("Starting transcription stream...")
        transcribed_chunks = []
        last_processed = 0
        chunks_needed = int(chunk_duration * self.sample_rate / 1024)

        model = WhisperModel(
            "large-v3",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8",
            download_root=str(
                Path.home() / "Library/Application Support/MacWhisper/models"
            ),
        )

        pbar = tqdm(desc="Transcribing", unit=" chunks", delay=1.0)

        while self.is_recording or last_processed < len(self.recording_chunks):
            current_chunks = len(self.recording_chunks)
            chunks_available = current_chunks - last_processed

            if (
                not self.is_recording
                and chunks_available > 0
                and chunks_available < chunks_needed
            ):
                chunks_needed = chunks_available

            if chunks_available >= chunks_needed:
                try:
                    # Process the next chunk
                    chunk_end = last_processed + chunks_needed
                    chunk_data = np.concatenate(
                        self.recording_chunks[last_processed:chunk_end]
                    )

                    # Save temporary chunk for transcription
                    temp_path = self.output_dir / f"temp_chunk_{last_processed}.wav"
                    with wave.open(str(temp_path), "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(self.sample_rate)
                        wf.writeframes((chunk_data * 32767).astype(np.int16).tobytes())

                    segments, _ = model.transcribe(
                        str(temp_path), language="nl", beam_size=5
                    )
                    transcript = " ".join([segment.text for segment in segments])

                    if transcript.strip():
                        print(
                            f"\nTranscribed chunk {last_processed//chunks_needed + 1}:"
                        )
                        print(f"{transcript}\n")
                        transcribed_chunks.append(transcript)  # Store the transcript

                    # Cleanup
                    temp_path.unlink()
                    last_processed = chunk_end

                    # Add more detailed logging
                    self.logger.info(
                        f"Successfully processed chunk {last_processed//chunks_needed + 1}"
                    )
                    self.logger.debug(f"Transcript length: {len(transcript)}")

                    pbar.update(1)  # Update progress bar na elke chunk

                except Exception as e:
                    self.logger.error(f"Error processing chunk: {e}", exc_info=True)
                    continue  # Continue instead of break to try processing remaining chunks
            elif self.is_recording:
                sd.sleep(500)

        pbar.close()
        self.logger.info(
            f"Transcription complete. Processed {len(transcribed_chunks)} chunks"
        )
        return transcribed_chunks

    def extract_activities(self, transcript: str) -> List[str]:
        """Generate a structured meeting summary using Ollama LLM.

        Processes the meeting transcript in two stages:
        1. Extracts a concise subject (2-3 words) for filename
        2. Generates a detailed structured summary

        The summary follows a specific format:
        - Context (participants, time, location)
        - Main points (3-5 key topics)
        - Details (discussions, decisions, insights)
        - Action items (follow-up tasks)

        Args:
            transcript (str): Complete meeting transcript

        Returns:
            List[str]: Structured summary split by lines

        Side Effects:
            - Saves summary to file: {timestamp}-{subject}.txt
            - Logs summary generation status
        """
        self.logger.info("Generating meeting summary using Ollama...")
        client = Client(host="http://localhost:11434")

        # Updated subject prompt to explicitly forbid commas
        subject_prompt = """
        Geef in 2-3 woorden het hoofdonderwerp van dit gesprek.
        Gebruik alleen letters, cijfers en koppeltekens (geen spaties, komma's of andere tekens).
        
        Gesprek:
        {transcript}
        """

        subject_response = client.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": subject_prompt.format(transcript=transcript),
                }
            ],
        )

        # Clean the subject to ensure no commas
        subject = (
            (
                subject_response.get("message", {}).get("content")  # New format
                or subject_response.get("content")  # Old format
                or "algemeen-gesprek"  # Fallback
            )
            .strip()
            .replace(",", "-")
        )

        # Original summary prompt...
        prompt = """
        Analyseer het volgende Nederlandse gesprek en maak een gestructureerde samenvatting.
        Geef de samenvatting in het Nederlands.
        
        Format de samenvatting als volgt:
        
        CONTEXT:
        - Wie waren aanwezig
        - Wanneer en waar vond het gesprek plaats (indien genoemd)
        
        HOOFDPUNTEN:
        - Belangrijkste besproken onderwerpen (max 3-5 punten)
        
        DETAILS:
        - Relevante details en discussiepunten
        - Genomen beslissingen
        - Belangrijke inzichten
        
        ACTIEPUNTEN:
        - Concrete vervolgacties (indien genoemd)
        
        Als er geen duidelijke inhoud is gevonden voor een sectie, gebruik dan "Niet genoemd" of "Geen".
        
        Gesprek:
        {transcript}
        """

        response = client.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt.format(transcript=transcript)}
            ],
        )

        message_content = (
            response.get("message", {}).get("content")  # New format
            or response.get("content")  # Old format
            or "Geen samenvatting beschikbaar."  # Fallback
        )

        # Update timestamp format to exclude seconds
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        summary_path = self.output_dir / f"{timestamp}-{subject}.txt"

        try:
            summary_path.write_text(message_content)
            self.logger.info(f"Summary saved to {summary_path}")
        except Exception as e:
            self.logger.error(f"Failed to save summary: {e}")

        return message_content.split("\n")


def main():
    """Entry point for the meeting recording application.

    Orchestrates the recording process through multiple threads:
    1. Recording Thread: Captures audio input
    2. Transcription Thread: Processes audio chunks

    Flow:
    1. Initializes recorder and waits for user input
    2. Starts parallel recording and transcription
    3. Monitors for Ctrl+C to stop recording
    4. Processes remaining audio and generates summary
    5. Cleans up temporary audio files

    Error Handling:
    - Graceful shutdown on keyboard interrupt
    - Timeout protection for transcription
    - Automatic cleanup of temporary files
    - Comprehensive error logging
    """
    recorder = MeetingRecorder()
    audio_file_path = None

    print("Press Enter to start recording, and Ctrl+C to stop...")
    input()

    transcribed_text = []
    stop_event = threading.Event()

    def recording_target():
        try:
            nonlocal audio_file_path
            audio_file_path = recorder.start_recording()
        except Exception as e:
            logging.error(f"Recording error: {e}", exc_info=True)
            stop_event.set()

    def transcription_target():
        try:
            chunks = recorder.transcribe_audio_stream()
            transcribed_text.extend(chunks)
        except Exception as e:
            logging.error(f"Transcription error: {e}", exc_info=True)
            stop_event.set()

    # Start recording in separate threads
    recording_thread = threading.Thread(target=recording_target)
    transcription_thread = threading.Thread(target=transcription_target)

    recording_thread.start()
    transcription_thread.start()

    print("Recording in progress. Press Ctrl+C to stop...")
    with tqdm(desc="Recording", unit=" seconds") as pbar:
        try:
            while recording_thread.is_alive():
                recording_thread.join(timeout=1)
                pbar.update(1)
        except KeyboardInterrupt:
            print("\nStopping recording...")
            recorder.stop_recording()
            stop_event.set()

        # Process any remaining audio
        print("Processing final chunks...")
        try:
            transcription_thread.join(timeout=30)  # Increased timeout to 30 seconds
        except TimeoutError:
            print("Transcription timed out")

    # Generate summary if we have transcribed text
    if transcribed_text:
        full_transcript = " ".join(transcribed_text)
        print("\nFull Transcript:")
        print(full_transcript)
        print("\nGenerating meeting summary...")
        summary = recorder.extract_activities(full_transcript)
        print("\nMeeting Summary:")
        print("\n".join(summary))
    else:
        print("\nNo transcribed text was captured!")

    # After generating summary, delete the audio file
    if audio_file_path and audio_file_path.exists():
        try:
            audio_file_path.unlink()
            print(f"\nDeleted audio file: {audio_file_path}")
        except Exception as e:
            print(f"\nFailed to delete audio file: {e}")

    print("\nRecording and transcription complete!")


if __name__ == "__main__":
    main()

import logging
import tempfile
import os
from pathlib import Path
from subprocess import run, PIPE, CalledProcessError

from pydub import AudioSegment

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider
from audiobook_generator.utils.utils import set_audio_tags

logger = logging.getLogger(__name__)

class XTTSTTSProvider(BaseTTSProvider):
    """High-quality local TTS provider using XTTS (Coqui's cross-lingual TTS)"""
    
    def __init__(self, config: GeneralConfig):
        config.output_format = config.output_format or "mp3"
        self.price = 0.000  # Local TTS is free
        super().__init__(config)

    def __str__(self) -> str:
        return f"XTTSTTSProvider(config={self.config})"

    def validate_config(self):
        """Validate XTTS configuration"""
        # Check if xtts is installed
        try:
            result = run(["python", "-c", "import TTS"], capture_output=True, text=True)
            if result.returncode != 0:
                raise ValueError("XTTS (TTS library) is not installed. Please install with: pip install TTS")
        except FileNotFoundError:
            raise ValueError("Python not found in PATH")
        
        # Check for required voice file if specified
        if hasattr(self.config, 'xtts_voice_file') and self.config.xtts_voice_file:
            if not Path(self.config.xtts_voice_file).exists():
                raise ValueError(f"XTTS voice file not found: {self.config.xtts_voice_file}")

    def text_to_speech(self, text: str, output_file: str, audio_tags: AudioTags):
        """Convert text to speech using XTTS"""
        logger.info("Starting XTTS text-to-speech conversion")
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            logger.debug("Created temporary directory %r", tmpdirname)
            
            # Generate unique temporary filenames
            tmp_wav_file = Path(tmpdirname) / "xtts_output.wav"
            tmp_input_file = Path(tmpdirname) / "input.txt"
            
            # Write text to temporary file
            with open(tmp_input_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Prepare XTTS command
            cmd = [
                "python", "-c", self._get_xtts_script(),
                "--text_file", str(tmp_input_file),
                "--output_file", str(tmp_wav_file),
                "--language", getattr(self.config, 'xtts_language', 'en'),
                "--model_name", getattr(self.config, 'xtts_model', 'tts_models/multilingual/multi-dataset/xtts_v2')
            ]
            
            # Add speaker/voice file if specified
            if hasattr(self.config, 'xtts_voice_file') and self.config.xtts_voice_file:
                cmd.extend(["--speaker_wav", self.config.xtts_voice_file])
            elif hasattr(self.config, 'xtts_speaker') and self.config.xtts_speaker:
                cmd.extend(["--speaker_idx", str(self.config.xtts_speaker)])
            
            # Add speed parameter if specified
            if hasattr(self.config, 'xtts_speed') and self.config.xtts_speed:
                cmd.extend(["--speed", str(self.config.xtts_speed)])
            
            logger.info(f"Running XTTS command: {' '.join(cmd[:3])} [arguments hidden for brevity]")
            
            try:
                result = run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    logger.error(f"XTTS command failed with return code {result.returncode}")
                    logger.error(f"STDOUT: {result.stdout}")
                    logger.error(f"STDERR: {result.stderr}")
                    raise CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
                    
                logger.info("XTTS synthesis completed successfully")
                
            except Exception as e:
                logger.error(f"Error running XTTS: {e}")
                raise
            
            # Verify output file was created
            if not tmp_wav_file.exists():
                raise FileNotFoundError(f"XTTS did not create output file: {tmp_wav_file}")
            
            # Set audio tags before conversion
            if audio_tags:
                set_audio_tags(tmp_wav_file, audio_tags)
            
            logger.info(f"Converting XTTS output to {self.config.output_format} format")
            
            # Convert to desired output format
            audio_segment = AudioSegment.from_wav(tmp_wav_file)
            audio_segment.export(output_file, format=self.config.output_format)
            
            logger.info(f"Conversion completed, output file: {output_file}")

    def _get_xtts_script(self):
        """Return the Python script for XTTS synthesis"""
        return '''
import argparse
import torch
from TTS.api import TTS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--model_name", default="tts_models/multilingual/multi-dataset/xtts_v2")
    parser.add_argument("--speaker_wav", default=None)
    parser.add_argument("--speaker_idx", type=int, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Read text from file
    with open(args.text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Initialize XTTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(args.model_name).to(device)
    
    # Generate speech
    if args.speaker_wav:
        # Voice cloning mode
        tts.tts_to_file(
            text=text,
            speaker_wav=args.speaker_wav,
            language=args.language,
            file_path=args.output_file,
            speed=args.speed
        )
    elif args.speaker_idx is not None:
        # Multi-speaker mode
        tts.tts_to_file(
            text=text,
            speaker=args.speaker_idx,
            language=args.language,
            file_path=args.output_file,
            speed=args.speed
        )
    else:
        # Single speaker mode
        tts.tts_to_file(
            text=text,
            language=args.language,
            file_path=args.output_file,
            speed=args.speed
        )

if __name__ == "__main__":
    main()
'''

    def estimate_cost(self, total_chars):
        return 0  # XTTS is free (local processing)

    def get_break_string(self):
        return "."  # Period as default break string

    def get_output_file_extension(self):
        return self.config.output_format


def get_xtts_supported_languages():
    """Return list of languages supported by XTTS"""
    return [
        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", 
        "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"
    ]

def get_xtts_supported_models():
    """Return list of XTTS models available"""
    return [
        "tts_models/multilingual/multi-dataset/xtts_v2",
        "tts_models/en/ljspeech/xtts_v2",
        "tts_models/en/vctk/xtts_v2"
    ]

def get_xtts_supported_output_formats():
    """Return list of output formats supported by XTTS"""
    return ["wav", "mp3", "flac", "ogg"]
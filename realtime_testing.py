import subprocess
from streamingxtts import Model

# Example usage
if __name__ == "__main__":

    def stream_ffplay(audio_stream, output_file=None):
        if output_file is None:
            ffplay_cmd = ["ffplay", "-nodisp", "-autoexit", '-f', 's16le', '-ar', '24000', '-ac', '1', "-"]
        else:
            print("Saving to", output_file)
            ffplay_cmd = ["ffmpeg", "-y", "-f", "wav", "-i", "-", output_file]

        with subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE) as ffplay_proc:
            try:
                for chunk in audio_stream:
                    ffplay_proc.stdin.write(chunk)
            except BrokenPipeError:
                pass  # Handle the case where ffplay ends prematurely
            except Exception as e:
                print(f"Unexpected error: {e}")
            finally:
                ffplay_proc.stdin.close()
                ffplay_proc.wait()

    model = Model()
    model.load()
    model_input = {
        "text": """In Verdant Valley... Elara[laughter], a young cartographer, charted forgotten trails. Her maps told stories of winding rivers and starlit glades. One autumn evening, 
        she found a glowing parchment in a hollow oak... Its frayed edges pulsed with strange symbols, marking Aeloria, the Lost Realm. Legends whispered of Aeloria’s treasures: 
        singing crystals and rivers of starlight. Elara’s heart raced with curiosity. She vowed to uncover its secrets, knowing the journey would test her courage. 
        Under the moon’s glow, she began her quest, her map a beacon in the night.""",
        "language": "en",
        "chunk_size": 20
    }
    stream_ffplay(model.predict(model_input), None)
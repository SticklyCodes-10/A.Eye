import os
import cv2
import base64
from groq import Groq
from gtts import gTTS
from pydub import AudioSegment
import tempfile
import simpleaudio as sa

# Your Groq API key
API_KEY = "YOUR_GROQ_API_KEY"

# Initialize Groq client with the API key
client = Groq(api_key=API_KEY)

# Function to capture an image from the webcam
def capture_image(image_path):
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return False

    ret, frame = cap.read()  # Capture a frame
    if ret:
        cv2.imwrite(image_path, frame)  # Save the image
        print(f"Image captured and saved as {image_path}")
    else:
        print("Error: Could not read frame.")

    cap.release()  # Release the camera
    return ret

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def describe_image(image_path):
    try:
        # Getting the base64 string
        base64_image = encode_image(image_path)

        # Create chat completion request
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="llava-v1.5-7b-4096-preview",
            temperature=0,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        # Get output message
        output_message = chat_completion.choices[0].message.content
        return output_message

    except Exception as e:
        return f"An error occurred: {e}"

# Function to convert text to speech and play it
def speak(text):
    try:
        # Create a temporary file for the audio
        tts = gTTS(text)
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tts.save(temp_file.name)

        # Load and play the audio with pydub/simpleaudio
        audio = AudioSegment.from_mp3(temp_file.name)
        play_obj = sa.play_buffer(audio.raw_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)
        play_obj.wait_done()
        
        os.remove(temp_file.name)
    except Exception as e:
        print(f"An error occurred during TTS: {e}")

def main():
    image_path = "pic.jpg"
    
    # Capture an image
    if capture_image(image_path):
        output_message = describe_image(image_path)  # Get response from the API
        print("Image Description:", output_message)  # Print the image description

        # Convert description to speech if available
        if "An error occurred" not in output_message:
            speak(output_message)

        # Remove the image file
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted image file: {image_path}")

if __name__ == "__main__":
    main()

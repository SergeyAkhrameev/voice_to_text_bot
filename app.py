import os
import time
import numpy as np
import json
import torch

import telebot as tb
import whisper_timestamped as whisper
import ffmpeg
import math

from pydub import AudioSegment
from dotenv import load_dotenv



load_dotenv()

#Provide the following values to the .env file
TG_BOT_TOKEN = os.getenv('TG_BOT_TOKEN')


#this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())

#help(whisper.transcribe)

bot = tb.TeleBot('TG_BOT_TOKEN')

# Voice messages processor
@bot.message_handler(content_types=["voice"])
def handle_voice_message(message):
    try:
        # Getting a voice message
        voice = bot.get_file(message.voice.file_id)
        voice_info = bot.get_file(message.voice.file_id)
        voice_file = bot.download_file(voice.file_path)

        # Saving a voice message to a file
        voice_file_path_ogg = os.path.join("voice_messages", f"{message.message_id}.ogg")
        with open(voice_file_path_ogg, "wb") as voice_file_local:
            voice_file_local.write(voice_file)

        # Converting .ogg to .wav
        voice_file_path_wav = os.path.join("voice_messages", f"{message.message_id}.wav")
        audio = AudioSegment.from_ogg(voice_file_path_ogg)
        audio = audio.set_frame_rate(16000).set_sample_width(2)  # Set up desired frequency 
        audio.export(voice_file_path_wav, format="wav")

        #voice_file_path_wav ='voice_messages/respondent.wav' #Load file directly

        print(f"voice_file_path_wav: {voice_file_path_wav}")

        # Speach detection with WhisperTimestamped
        audio = whisper.load_audio(voice_file_path_wav)
        model = whisper.load_model("large", device='cpu')
        
        start_time = time.time()
        result = whisper.transcribe(model, audio, language="ru")
        end_time = time.time()

        elapsed_time = end_time - start_time
        

        #print(torch.device)



        '''
        audio_list=[]
        for i in range(len(result['text']['segments'][i])):
            audio_data = dict(audio_id = result['text']['segments'][i]['start']['text'],)
                
            audio_list.append(audio_data)

            return audio_list
        '''
        
        json_result = json.dumps(result, indent = 2, ensure_ascii = False)
        print(json_result)

        segments =[]
        for item in result['segments']:
            confidence = item['confidence']
            segments.append(confidence)
        
        # Calculate an averege confidence
        if segments:
            average_confidence = np.mean(segments)
            print("Average confidence:", average_confidence)
        else:
            print("The segments list is empty.")

        with open("result.txt", "w", encoding="utf-8") as text_file:
            text_file.write(json_result)

        # Max message length in Telegram
        MAX_MESSAGE_LENGTH = 4000

        

        # Sending text
        text_to_send = result['text']

        if len(text_to_send) <= MAX_MESSAGE_LENGTH:
            bot.send_message(message.chat.id, f"Recognized text: {result['text']}")

        else:

            # Splitting text into words
            words = text_to_send.split()

            # Initializing a list of the message parts
            message_parts = []

            current_part = ""
            for word in words:
                if len(current_part) + len(word) + 1 <= MAX_MESSAGE_LENGTH:
                    # Adding a word and a space
                    current_part += word + " "
                else:
                    # If limit is reached, stop appending
                    message_parts.append(current_part.strip())
                    current_part = word + " "

            # Adding the last part
            message_parts.append(current_part.strip())

            # Sending the parts to the bot
            for part in message_parts:
                bot.send_message(message.chat.id, f"Recognized text: {part}")    


    
        print(f"Completion time is: {elapsed_time} seconds")
    
    except Exception as e:
        bot.send_message(message.chat.id, f"Error occured: {str(e)}")

# Run the bot
bot.polling(none_stop=True)
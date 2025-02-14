import azure.functions as func
import logging
#Azure AI Cognitive Services
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
#from azure.cognitiveservices.speech import SpeechRecognizer
import azure.cognitiveservices.speech as speechsdk
from azure.ai.translation.document import DocumentTranslationClient
from azure.core.credentials import AzureKeyCredential
from nltk.translate.bleu_score import sentence_bleu
app = func.FunctionApp()
reference = [
    'this is a dog'.split(),
    'it is dog'.split(),
    'dog it is'.split(),
    'a dog, it is'.split() 
]
print(reference)
@app.blob_trigger(arg_name="myblob", path="Samples/{blobname}.{blobextension}",
                               connection="https://azaitranslation.blob.core.windows.net/azaicogsvcstorage/") 
def azaitranslation(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob"
                f"Name: {myblob.name}"
                f"Blob Size: {myblob.length} bytes")
    image_text = agent_image_ocr(myblob.name)
    audio_text = agent_speech_to_text(myblob.name)
    translated_text = agent_translate_text(myblob.name, "es")
    
    print(image_text)
    print('BLEU score -> {}'.format(sentence_bleu(reference, image_text)))
    print(audio_text)
    print('BLEU score -> {}'.format(sentence_bleu(reference, audio_text)))
    print(translated_text)
    print('BLEU score -> {}'.format(sentence_bleu(reference, translated_text)))

region="East US"
# Initialize clients
vision_client = ComputerVisionClient("https://eastus.api.cognitive.microsoft.com/", AzureKeyCredential("44c9283d2e014e5389363e4b35d37123"))
#speech_client = SpeechRecognizer("https://eastus.api.cognitive.microsoft.com/", AzureKeyCredential("e86739329ba54317aea3c0a903c5b461"))
translation_client = DocumentTranslationClient("https://eastus.api.cognitive.microsoft.com/", AzureKeyCredential("6a70cd5bf5ed4b998b4a7b8f6d5e542c"))

vision_client = ComputerVisionClient("https://eastus.api.cognitive.microsoft.com/", AzureKeyCredential("44c9283d2e014e5389363e4b35d37123"))
#speech_config = SpeechConfig(endpoint="https://eastus.api.cognitive.microsoft.com/",auth_token=AzureKeyCredential("e86739329ba54317aea3c0a903c5b461"))
#speech_recognizer = SpeechRecognizer(speech_config=speech_config)
speech_config = speechsdk.SpeechConfig(
        subscription="e86739329ba54317aea3c0a903c5b461", region=region)

speech_config.request_word_level_timestamps()
speech_config.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceResponse_OutputFormatOption, value="detailed")

    # Creates a speech recognizer using the default microphone (built-in).
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config)
translation_client = DocumentTranslationClient("https://eastus.api.cognitive.microsoft.com/", AzureKeyCredential("6a70cd5bf5ed4b998b4a7b8f6d5e542c"))

# Agent 1: Image OCR
def agent_image_ocr(image_path):
    with open(image_path, "rb") as image_stream:
        ocr_result = vision_client.recognize_text(image_stream)
    return ocr_result

# Agent 2: Speech-to-Text
def agent_speech_to_text(audio_path):
    result = speech_recognizer.recognize_speech(audio_path)
    return result.text

# Agent 3: Translator
def agent_translate_text(text, target_language):
    response = translation_client.translate(text, target_language)
    return response.translations[0].text

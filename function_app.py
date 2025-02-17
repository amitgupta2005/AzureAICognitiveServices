import os
import azure.functions as func
import logging
#Azure AI Cognitive Services
#from azure.cognitiveservices.vision.computervision import ComputerVisionClient
#from azure.cognitiveservices.speech import SpeechRecognizer
#import azure.cognitiveservices.speech as speechsdk
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from nltk.translate.bleu_score import sentence_bleu

from azure.core.credentials import AzureKeyCredential
from azure.ai.translation.text import TextTranslationClient

app = func.FunctionApp()
#endpoint="https://api.cognitive.microsofttranslator.com/"
endpoint="https://azaitexttranslator.cognitiveservices.azure.com/"
api_key="EaJZfppSjy2KCqL7DvjNJi5GiVgqOkWH8VLSjAfkFbmnRvF8PVtEJQQJ99BBACYeBjFXJ3w3AAAbACOG9bwy"
credential=AzureKeyCredential(api_key)
client=TextTranslationClient(endpoint=endpoint,credential=credential)
reference = [
    'Esto es un perro'.split(),
    'C’est un chien'.split(),
    'Das ist ein Hund'.split(),
    #'this is a dog'.split() 
]
def translate_text(text,target_languages):
    result = {}
    for language in target_languages:
        response=client.translate(body=[text], to_language=[language])
        result[language]=response[0].translations[0].text
    return result
def main():
    text_to_translate="this is a dog"
    target_languages=["es","fr","de"]
    translated_text=translate_text(text_to_translate,target_languages)
    print(translated_text)
    print('BLEU score -> {}'.format(sentence_bleu(reference, translated_text)))

if __name__ == "__main__":
    main()
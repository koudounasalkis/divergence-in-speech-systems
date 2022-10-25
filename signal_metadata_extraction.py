from pydub import AudioSegment, silence
import string
import re

class MetadataExtractor:
    def __init__(self):
        self.list_signal_metadata = ['total_silence', 'total_duration', 'trimmed_duration']
        self.list_text_metadata = ['n_words']
        self.list_sig_text_metadata = ['speed_rate_word', 'speed_rate_word_trimmed']

        
    def signal_metadata(self, audio_file):
        try:
          if audio_file.endswith('.mp3') or audio_file.endswith('.MP3'):
            audio_seg = AudioSegment.from_mp3(audio_file)
          elif audio_file.endswith('.wav') or audio_file.endswith('.WAV'):
            audio_seg = AudioSegment.from_wav(audio_file)
          elif audio_file.endswith('.ogg'):
            audio_seg = AudioSegment.from_ogg(audio_file)
          elif audio_file.endswith('.flac'):
            audio_seg = AudioSegment.from_file(audio_file, "flac")
          elif audio_file.endswith('.3gp'):
            audio_seg = AudioSegment.from_file(audio_file, "3gp")
          elif audio_file.endswith('.3g'):
            audio_seg = AudioSegment.from_file(audio_file, "3gp")
        except:
          print("Couldn't load file")
          return None
        
        # Total length of the silence
        silences, total_silence = get_silences(audio_seg, verbose = False)
        
        # Audio duration
        total_duration =  len(audio_seg)/1000 # in s
        
        # Speech duration without pauses
        trimmed_duration = get_trimmed_duration(audio_seg, total_duration) 
                                                
        # Metadata:
        ## Total silence/pauses        
        ## Total duration 
        ## Speech duration without all pauses
        
        return [total_silence, total_duration, trimmed_duration]
    

    def text_metadata(self, sentence):
        sentence = clear_sentence(sentence)

        # Metadata:
        ## Number of Words
        
        return [len(sentence.split(' '))]
    

    def mixed_metadata(self, signal_metadata, text_metadata):

        # Metadata:
        ## Speaking Word Rate 
        ## Spearking Word Rate Trimmed

        total_duration = signal_metadata[self.list_signal_metadata.index('total_duration')]
        trimmed_duration = signal_metadata[self.list_signal_metadata.index('trimmed_duration')]
        n_words = text_metadata[self.list_text_metadata.index('n_words')]
        
        return [total_duration / n_words, trimmed_duration / n_words]
    
    
def get_silences(myaudio, min_silence_len=150, verbose = False):
    """
    Input:
    myaudio: AudioSegment or  path of the audio (audio_file)
    min_silence_len : minimum length of the 'silence' (pause) in ms
    
    Output:
    List of pauses: [(0.0, 0.695), (1.737, 3.157)]
    Total length of pauses/silence
    """ 
    
    if type(myaudio) == str: 
        try:
          if myaudio.endswith('.mp3') or myaudio.endswith('.MP3'):
            myaudio = AudioSegment.from_mp3(myaudio)
          elif myaudio.endswith('.wav') or myaudio.endswith('.WAV'):
            myaudio = AudioSegment.from_wav(myaudio)
          elif myaudio.endswith('.ogg'):
            myaudio = AudioSegment.from_ogg(myaudio)
          elif myaudio.endswith('.flac'):
            myaudio = AudioSegment.from_file(myaudio, "flac")
          elif myaudio.endswith('.3gp'):
            myaudio = AudioSegment.from_file(myaudio, "3gp")
          elif myaudio.endswith('.3g'):
            myaudio = AudioSegment.from_file(myaudio, "3gp")
        except:
          print("Couldn't load file")
          return None, None   

    if type(myaudio) != AudioSegment:
        raise ValueError('Input should be a pydub AudioSegment or the path of the audio')

    dBFS=myaudio.dBFS

    silences = silence.detect_silence(myaudio, min_silence_len=min_silence_len, silence_thresh=dBFS-16)

    silences = [((start/1000),(stop/1000)) for start,stop in silences] #in sec
    if verbose:
        print(f'Silences length for min_silence_size (ms) {min_silence_len}', [(s[1]-s[0])*1000 for s in silences])
        print(f'Silences, start and stop: {silences}')
        
    total_silence = sum([(s[1]-s[0]) for s in silences])
    
    return silences, total_silence


def get_trimmed_duration(myaudio, total_duration, min_silence_len = 50):
    """
    Input:
    audio_seg: AudioSegment or  path of the audio (audio_file)
    min_silence_len : minimum length of the 'silence' (pause) in ms --> keep it short (e.g., 50)
    to identify the first and last 'silence' and remove it.
    """
    
    silences, _ = get_silences(myaudio, min_silence_len = min_silence_len)
    start, end = 0, total_duration
    
    if (len(silences) > 0):
      if silences[0][0] == 0:
          start = silences[0][1] 
      if len(silences)>1 and silences[-1][1]: 
          end = silences[-1][0]

    return end-start


def clear_sentence(sentence):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation)).strip()
    return re.sub(' +',' ',sentence)
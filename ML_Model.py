# Import Libraries

import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
import io
nltk.download('punkt')


# Load data============================================================================================

Data = pd.read_csv("DataFiles/Sinhala_Singlish_Hate_Speech.csv")


# Data Handling Techniques=============================================================================

# Create DataFrame----------------------


df_Data = pd.DataFrame(columns=['PhraseNo', 'Phrase', 'IsHateSpeech'])
for i in range(Data.shape[0]):
    df_Data.loc[i] = [i + 1] + [Data['Phrase'][i]] + [Data['IsHateSpeech'][i]]


# Sinhala Stop Words Removal----------------------

# getting the stop words

sinhala_stop_Words = io.open("DataFiles/StopWords_425.txt",mode="r", encoding="utf-16")
stop_words_Sinhala = []
for x in sinhala_stop_Words:
    stop_words_Sinhala.append(x.split()[0])


AllDataSinhala = df_Data

for k in range(AllDataSinhala.shape[0]):
    sinhala_tokenize_words = word_tokenize(str(AllDataSinhala['Phrase'][k]))

    # remove stop words
    removing_Sinhala_stopwords = set(sinhala_tokenize_words) - set(stop_words_Sinhala)

    # print(removing_stopwords_sentence)
    df_Data.loc[k] = [k + 1] + [removing_Sinhala_stopwords] + [AllDataSinhala['IsHateSpeech'][k]]


# Romanized Sinhala Stop Words removal-----------------------------------

# getting the stop words
singlish_stop_Words = io.open("DataFiles/Singlish StopWords_425.txt", mode="r",
    encoding="utf-8")
stop_words_singlish = []
for y in singlish_stop_Words:
    stop_words_singlish.append(y.split()[0])

AllDataSinglish = df_Data

for r in range(AllDataSinglish.shape[0]):
    siglish_tokenize_words = word_tokenize(str(AllDataSinglish['Phrase'][r]))

    # remove stop words
    removing_Singlish_stopwords = set(siglish_tokenize_words) - set(stop_words_singlish)

    # print(removing_stopwords_sentence)
    df_Data.loc[r] = [r + 1] + [removing_Singlish_stopwords] + [AllDataSinglish['IsHateSpeech'][r]]


# Sinhala Steming---------------------------------------------------

suffixes = io.open("DataFiles/Suffixes-413.txt", mode="r",encoding="utf-16")
sinhala_suffixes = []

for suf in suffixes:
    sinhala_suffixes.append(suf.strip().split()[0])

sentences = nltk.sent_tokenize(str(df_Data['Phrase']))

for j in range(len(sentences)):
    sentence_tokenize = nltk.sent_tokenize(sentences[j])

    stem_sentences = set(sentence_tokenize) - set(sinhala_suffixes)

    # print(lemmatized_words)
    df_Data.loc[j] = [j + 1] + [stem_sentences] + [df_Data['IsHateSpeech'][j]]



# Data Cleaning------------------------------------------------


def TagsRemovalText(sentence):
    # regex for html tags cleaner
    htmlTags = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext_htmltags = re.sub(htmlTags, '', sentence).lower()  # convert to lower case

    # regex for non alphabetical characters cleaner
    cleantext_NonAlp = re.compile(u'[^\u0061-\u007A|^\u0D80-\u0DFF|^\u0030-\u0039]', re.UNICODE)
    # Englosh lower case unicode range = \u0061-\u007A
    # Sinhala unicode range = |u0D80-\u0DFF
    # Numbers unicode range = \u0030-\u0039

    cleantext_finalText = re.sub(cleantext_NonAlp, ' ', cleantext_htmltags).strip(" ")

    # tokenzing
    # finalText = word_tokenize(cleantext_finalText)
    # finalText = sent_token = nltk.sent_tokenize(tokenzie_finalText)

    # return finalText
    return cleantext_finalText


for i in range(df_Data.shape[0]):
    dataSentence = df_Data['Phrase'][i]
    preprocessData = TagsRemovalText(str(dataSentence))
    df_Data.loc[i] = [i + 1] + [preprocessData] + [df_Data['IsHateSpeech'][i]]


# Sinhala Steming------------------------------------------------------

suffixes = io.open("DataFiles/Suffixes-413.txt", mode="r",encoding="utf-16")
sinhala_suffixes = []
afterStemingLenghtsUnique = []

for suf in suffixes:
    sinhala_suffixes.append(suf.strip().split()[0])


def isSuffix(s1, s2):
    n1 = len(s1)
    n2 = len(s2)
    if (n1 > n2):
        return False
    for i in range(n1):
        if (s1[n1 - i - 1] != s2[n2 - i - 1]):
            return False
    return True


def removeSuffix(word, suffix):
    newLen = len(word) - len(suffix)
    wordN = word[0:newLen]
    return wordN


def Data_stemming(stemming):
    stems = {}
    found = 0
    for r in range(stemming.shape[0]):
        Sentence_stem = stemming['Phrase'][r]

        SentenceTokens_stem = word_tokenize(Sentence_stem)
        stemming_sentence_n = []
        for wr in SentenceTokens_stem:
            found = 0
            for suf in sinhala_suffixes:
                if (isSuffix(suf.strip(), wr.strip())):
                    stm = removeSuffix(wr.strip(), suf.strip())
                    stems[wr] = stm
                    stemming_sentence_n.append(stems[wr])
                    found = 1
                    break

            if (found == 0):
                stemming_sentence_n.append(wr)

        # load to dataframe
        df_Data.loc[r] = [r + 1] + [stemming_sentence_n] + [stemming['IsHateSpeech'][r]]
    return stemming


Data_stemming(df_Data)
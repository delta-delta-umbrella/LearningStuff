"""
Script to load the json files of COVID-19 dataset, obtain the raw text and save the dataset

"""
import os
import json
import pandas

class DataLoader():

    def ObtainData(self, directory):
        df = pandas.DataFrame(columns=['Text'])
        for filename in os.listdir(directory):
            if filename.endswith(".json") or filename.endswith(".py"):
                # print(os.path.join(directory, filename))
                json_filename = os.path.join(directory, filename)
                with open(json_filename) as json_file:
                    data = json.load(json_file)
                    for p in data['abstract']:
                        df = df.append({'Text': p['text']}, ignore_index=True)
                        # print(p['text'])
                    for q in data['body_text']:
                        df = df.append({'Text': q['text']}, ignore_index=True)
                        # print(p['text'])
                continue
            else:
                continue
        return df

    def SaveData(self, df, filename):
        df.to_csv(filename, encoding="utf-8")


# Path to dataset
directory = 'D:/Dan/PythonProjects/SciBERT_CORD19/RawText'

data = DataLoader()
savedir = data.ObtainData(directory)
data.SaveData(savedir, 'RawText.csv')

#print(RawText)

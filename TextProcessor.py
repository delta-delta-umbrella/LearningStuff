import pandas
import re
import codecs

class TextProcessor():

    def __init__(self, data):
        self.data = data

    def ProcessData(self):
        SENTENCE_REGEX = re.compile('[^!?\.]+[!?\.]')
        df = pandas.DataFrame(columns=['Processed Text'])
        len(self.data)  # Checking length of dataset
        for i in range(0, len(self.data)):

            if i % 5000 == 0:  # To see progress
                print(i)

            text = self.data['Text'].iloc[i]
            if len(text) > 26:
                modtext = re.sub("i\.e\.", 'i[PROTECTED_DOT]e[PROTECTED_DOT]', text)
                modtext = re.sub("i\.e", 'i[PROTECTED_DOT]e[PROTECTED_DOT]', modtext)
                modtext = re.sub("e\.g\.", 'e[PROTECTED_DOT]g[PROTECTED_DOT]', modtext)
                modtext = re.sub(r'(\b)eg\.', r'\1e[PROTECTED_DOT]g[PROTECTED_DOT]', modtext)
                modtext = re.sub(r'(\b)ex\.', r'\1ex[PROTECTED_DOT]', modtext)
                modtext = re.sub(r'\[[^)]*]', '', modtext)
                modtext = re.sub('http://\S+|https://\S+|\(http://\S+\)|\(https://\S+\)', '', modtext)
                modtext = re.sub('\w+\.html', '', modtext)
                modtext = re.sub('\w+\.com', '', modtext)
                modtext = re.sub('Figure \S+|Table \S+|\(Figure \S+\)|\(Table \S+\)|Fig\. \S+|\(Fig\. \S+\)|Figs\. \S+|\(Figs\. \S+\)|Fig \S+|\(Fig \S+\)', '', modtext)
                modtext = re.sub('\( {0,}\)', '', modtext)
                modtext = re.sub('doi:\S+', '', modtext)
                modtext = re.sub(r"(\d)\.(\d)|([A-Z])\.([A-Za-z])", r"\1[PROTECTED_DOT]\2", modtext)
                modtext = re.sub(r"\.(\d)", r"[PROTECTED_DOT]\1", modtext)
                modtext = re.sub("([A-Z])\. ([a-z])", r"\1[PROTECTED_DOT]\2", modtext)
                modtext = re.sub("E\.coli", "E[PROTECTED_DOT]coli", modtext)
                modtext = re.sub("E\. coli", "E[PROTECTED_DOT]coli", modtext)
                modtext = re.sub("e\.coli", "e[PROTECTED_DOT]coli", modtext)
                modtext = re.sub("e\. coli", "e[PROTECTED_DOT]coli", modtext)
                modtext = re.sub("et al\.", "et al[PROTECTED_DOT]", modtext)
                modtext = re.sub("inc\.", "inc[PROTECTED_DOT]", modtext)
                modtext = re.sub("Ltd\.", "Ltd[PROTECTED_DOT]", modtext)
                modtext = re.sub("Co\.", "Co[PROTECTED_DOT]", modtext)
                modtext = re.sub("Cat\.", "Cat[PROTECTED_DOT]", modtext)
                modtext = re.sub("no\.", "no[PROTECTED_DOT]", modtext)
                modtext = re.sub("No\.", "No[PROTECTED_DOT]", modtext)
                modtext = re.sub("etc\.", "etc[PROTECTED_DOT]", modtext)
                modtext = re.sub("pr\.", "pr[PROTECTED_DOT]", modtext)
                modtext = re.sub("sp\.", "sp[PROTECTED_DOT]", modtext)
                modtext = re.sub("vs\.", "vs[PROTECTED_DOT]", modtext)
                modtext = re.sub("spp\.", "spp[PROTECTED_DOT]", modtext)
                modtext = re.sub("lab\.", "lab[PROTECTED_DOT]", modtext)
                modtext = re.sub(r'(\b)([A-Za-z])\.(\S)', r'\1\2[PROTECTED_DOT]\3', modtext)
                textlist = [x.lstrip() for x in SENTENCE_REGEX.findall(modtext)]
                textlist = [line.replace("[PROTECTED_DOT]", ".") for line in textlist]
                for j in range(0, len(textlist)):
                    if len(textlist[j]) > 22:
                        df = df.append({'Processed Text': textlist[j]}, ignore_index=True)
        return df


def LoadData(filename):
    df = pandas.read_csv(filename, usecols=['Text'], encoding='utf-8')
    return df


def SaveData(filename, data):
    data.to_csv(filename, encoding='utf-8')


def SaveDataText(filename, data):
    f = codecs.open(filename, "w", "utf-8")
    for j in range(0, len(data)):
        f.write(data['Processed Text'].iloc[j]+" \r\n\n")

testdata = LoadData('TestItems2.csv')
testdata = TextProcessor(testdata)
testdata = testdata.ProcessData()

SaveData('Test_Text100.csv', testdata)
SaveDataText('Test_Text100.txt', testdata)

# might have to do the masking yourself

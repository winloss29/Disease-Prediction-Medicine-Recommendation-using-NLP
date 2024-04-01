import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from matplotlib import pyplot as plt
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import nltk
import nltk
df = pd.read_csv("C:/Users/Vijayvardhan reddy/Desktop/Symptom2Disease.csv")
df.drop("Unnamed: 0",axis=1,inplace=True)
x = df["text"]
y = df["label"]
def lower_case(x):
    for i in range(len(x)):
        x[i] = str(x[i]).lower()
def number_remove(x):
    for i in range(len(x)):
        x[i] = re.sub(r"\d+",'',x[i])
def punctuation(x):
    punct = str.maketrans('','',string.punctuation)
    for i in range(len(x)):
        x[i] = x[i].replace(","," ")
        x[i] = x[i].translate(punct)
def white_space(x):
    for i in range(len(x)):
        x[i] = " ".join(x[i].split())
def token(x):
    for i in range(len(x)):
        x[i] = word_tokenize(x[i])
def stopword(x):
    stop_words = set(stopwords.words('english'))
    for i in range(len(x)):
        list1 = []
        for j in range(len(x[i])):
            if x[i][j] not in stop_words:
                list1.append(x[i][j])
        x[i] = list1
wd =  WordNetLemmatizer()
def pos_create(x):
    for i in range(len(x)):
        list1 = []
        for j in range(len(x[i])):
            list2 = []
            list2.append(x[i][j])
            list1.append(nltk.pos_tag(list2))
        x[i] = list1
def pos_place(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j][1].startswith('J'):
                x[i][j][1] = wordnet.ADJ
            elif x[i][j][1].startswith('V'):
                x[i][j][1] = wordnet.VERB
            elif x[i][j][1].startswith('N'):
                x[i][j][1] = wordnet.NOUN
            elif x[i][j][1].startswith('R'):
                x[i][j][1] = wordnet.ADV
            else:         
                x[i][j][1] =  None
def convert(x):
    for i in range(len(x)):
        list1 = []
        for j in range(len(x[i])):
            l = str(x[i][j]).split()
            punctuation(l)
            list1.append(list(l))
        x[i] = list1
def lem(x):
    for i in range(len(x)):
        list1 = []
        for j in range(len(x[i])):
            if (x[i][j][1] == None):
                list1.append(x[i][j][0])
            else:
                list1.append(wd.lemmatize(x[i][j][0],x[i][j][1]))
        x[i] = " ".join(list1)
lower_case(x)
number_remove(x)
punctuation(x)
white_space(x)
token(x)
stopword(x)
pos_create(x)
convert(x)
pos_place(x)
lem(x)
cv = CountVectorizer()
X = cv.fit_transform(x)
X_train,X_test,Y_train,Y_test = train_test_split(X,y,random_state =0)
lr = LogisticRegression(C=24,max_iter=150)
lr.fit(X_train,Y_train)
lr.score(X_test,Y_test)*100
test =lr.predict(X_test)
cm = confusion_matrix(test,Y_test)
#sns.heatmap(cm,annot=True,fmt='g')
#plt.ylabel("Prediction",fontsize=20)
#plt.xlabel("Actual",fontsize=20)
mnb = MultinomialNB()
mnb.fit(X_train,Y_train)
mnb.score(X_test,Y_test)*100
test =mnb.predict(X_test)
cm = confusion_matrix(test,Y_test)
#sns.heatmap(cm,annot=True,fmt='g')
#plt.ylabel("Prediction",fontsize=20)
#plt.xlabel("Actual",fontsize=20)
def disease(t):
    lower_case(t)
    number_remove(t)
    punctuation(t)
    white_space(t)
    token(t)
    stopword(t)
    pos_create(t)
    convert(t)
    pos_place(t)
    lem(t)
    l = cv.transform(t)
    y=lr.predict(l)
    for i in y:
        return i
medicine_database = {"Psoriasis":["Adalimumab (Humira)","Etanercept (Enbrel)","Infliximab (Remicade)","Ustekinumab (Stelara)","Secukinumab (Cosentyx)","Ixekizumab (Taltz)"]
                     ,"Varicose Veins":["Ibuprofen","Naproxen","Diosmin","Aescin-containing gels","Warfarin"]
                     ,"Typhoid":["Ciprofloxacin","Ceftriaxone","Azithromycin","Trimethoprim-Sulfamethoxazole","Ofloxacin","Cefixime"]
                     ,"Chicken pox":["Antihistamines","Acetaminophen","Topical Calamine Lotion","acyclovir"]
                     ,"Impetigo":["Mupirocin","Retapamulin","Clindamycin","Dicloxacillin"]
                     ,"Dengue":["acetaminophen ","Contact Doctor"]
                     ,"Fungal Infection":["Miconazole","Griseofulvin","Ketoconazole Shampoo","Miconazole Powder"]
                     ,"Common Cold":["Acetaminophen","Pseudoephedrine","Loratadine","Guaifenesin"]
                     ,"Pneunomia":["Amoxicillin","Fluoroquinolones","acetaminophen","influenza"]
                     ,"diabetes":['Metformin',"Sulfonylureas","Meglitinides", "DPP-4 Inhibitors","Thiazolidinediones"]
                     ,"peptic ulcer disease":["Proton Pump Inhibitors" ,"H2 Blockers","Antacids",'Cytoprotective Agents']
                     ,"drug reaction":['Antihistamines','Corticosteroids','Epinephrine','anaphylaxis']
                     ,"Gastroesophageal_reflux_disease":['Prokinetic Agents','Alginate Antacids','H2 Blockers','Proton Pump Inhibitors ']
                     ,"allergy":['Cetirizine','Loratadine','Fexofenadine','Diphenhydramine','Phenylephrine','Fluticasone']
                     ,"Urinary tract_infections":['Nitrofurantoin','Ciprofloxacin','Levofloxacin','Ceftriaxone']
                     ,"malaria":['Artemether-lumefantrine','Chloroquine','Quinine','Mefloquine','Primaquine']
                     ,"jaundice":['Hepatitis A','Hepatitis B','Hepatitis C']
                     ,"Cervical spondylosis":['Ibuprofen','Naproxen','Acetaminophen','Gabapentin']
                     ,"migrain":['Sumatriptan','Rizatriptan','Eletriptan','Naproxen','Ibuprofen','Acetaminophen']
                     ,"hypertension":['Lisinopril','Enalapril','Losartan','Valsartan','Amlodipine','Chlorthalidone']
                     ,"bronchial asthama":['Albuterol','Levalbuterol','Formoterol','Budesonide','Ipratropium']
                     ,"acne":['Tretinoin','Adapalene','Clindamycin','Erythromycin','Doxycycline']
                     ,"arthritis":['Ibuprofen','Naproxen','Methotrexate','Sulfasalazine','Infliximab']
                     ,"Dimorphic hemorrhoids":['Acetaminophen','Hydrocortisone Cream','Witch Hazel','Psyllium']

}
def medicines(d):
    if d in medicine_database:
        x=medicine_database[d]
        c=1
        for i in x:
            print("{0}. {1}".format(c,i))
            c+=1
    else:
        return "Can't provide medicines for now. Please contact doctor"
print("You are suffering from>>>>",end=" ")
x=disease(["I frequently get swallowing issues and the sensation that food is getting stuck in my throat. I regularly belch and feel bloated. My aftertaste is unpleasant all the time"])
def print_bold(text):
    print("\033[1m" + text + "\033[0m")
print_bold(x)
print()
print("Medicines Required are:")
medicines(disease(["I frequently get swallowing issues and the sensation that food is getting stuck in my throat. I regularly belch and feel bloated. My aftertaste is unpleasant all the time"]))

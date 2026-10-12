"""Fetch and deterministically convert the pinned OpenML 37 source."""
import csv
import hashlib
import io
from pathlib import Path
import urllib.request
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
URL="https://openml.org/data/v1/download/37/diabetes.arff"
EXPECTED_SHA="4eddd5b2b64679e8888348e306520a393d6a28e1ddc9643cfb76fc5d912d6d40"
COLUMNS=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']

def convert(raw):
    text=raw.decode('utf-8')
    lines=text.splitlines()
    start=next(i for i,line in enumerate(lines) if line.strip().lower()=='@data')+1
    rows=list(csv.reader(line for line in lines[start:] if line.strip() and not line.lstrip().startswith('%')))
    frame=pd.DataFrame(rows,columns=COLUMNS)
    frame['Outcome']=frame.Outcome.map({'tested_positive':1,'tested_negative':0})
    for col in COLUMNS[:-1]: frame[col]=pd.to_numeric(frame[col])
    for col in ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','Age','Outcome']:frame[col]=frame[col].astype(int)
    if len(frame)!=768 or frame.Outcome.sum()!=268:raise ValueError('Unexpected source population')
    return frame

def download():
    raw=urllib.request.urlopen(URL,timeout=60).read()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED_SHA:raise ValueError('Source SHA-256 changed; review before replacing data')
    target=ROOT/'data';target.mkdir(exist_ok=True)
    frame=convert(raw)
    (target/'diabetes.arff').write_bytes(raw)
    frame.to_csv(target/'diabetes.csv',index=False,lineterminator='\n')
    print('Verified OpenML 37: 768 records, 268 positive outcomes.')

if __name__=='__main__':download()

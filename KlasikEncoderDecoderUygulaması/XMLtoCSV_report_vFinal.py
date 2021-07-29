#!/usr/bin/env python
# coding: utf-8

from xml.etree import ElementTree as ET
import os 
import csv
import pandas as pd

def xmlRaporuAl(report_name):
  tree = ET.parse(report_name)
  root = tree.getroot()
  return root

# Id sayısı kadar rapor olusturacağım için önce kaç Id olduğunu buluyorum.
def id_Bul():
  '''
  Bu fonksiyon raporların ilişkili olduğu görüntünün id sini bulmamıza yardımcı olur.
  '''
  my_Images = [] # Goruntuleri tutacak olan liste
  for img in root.getchildren(): # xml dosyası içinde ve root altındaki tüm tag'leri dolaş
    if img.tag == 'parentImage': # bu tagler'den parentImage eşit ise 
      img_id = list(img.attrib.values())[0] #attribute degerlerinin ilk elemanı bize raporun ait oldugu image id donduruyor
      my_Images.append(img_id) # bunu id'ler icin olusturdugum listeye ekle
  return my_Images 

def finding_Bul():
  '''
  Bu fonksiyon xml tipindeki raporlar içerisindeki findings kısmını bulmamıza yardımcı olur.
  '''
  findings = root.find('./MedlineCitation/Article/Abstract/AbstractText[@Label="FINDINGS"]')

  finding_text = findings.text
  if finding_text == None:
    finding_text = 'No Findings'
  return finding_text


def satirEkle(report_id,finding):
  '''
  Bu fonksiyon bize sadece id ve findinglerin oldugu csv dosyasını oluşturmamızda yardımcı olur.
  inputs : report_id - raporumuzun ilişkili olduğu resmin id degeri
           finding - raporlardaki bulgumuz yani asıl raporu olusturan kısımdır.
  '''
  rows = [] # Değerlerimizi tutacak olan satır.
  
  report_as_csv = open('report_lastfinal.csv','a',encoding='utf-8')

  #Değerler satırını oluştur ve csv içerisine yükle.
  rows.append(report_id)
  rows.append(finding)
  csv.writer(report_as_csv).writerow(rows)

  #close openin file .
  report_as_csv.close()


dir = '/dataset/all_reports/'

with open('reports.csv','w',encoding='utf-8') as r:
  columns = ['Id','Findings']
  csv.writer(r).writerow(columns)

for file_name in os.listdir(dir):
  if file_name.endswith('.xml'):    
    root = xmlRaporuAl(file_name)
    print(file_name+" is loaded.\n")
  else:
    continue
  try:  
    my_images_all_id = id_Bul()
    finding = finding_Bul()
    print("Report Id/s:",my_images_all_id,
          "\nFindings:",finding,"\n")
  except:
    print("Exception : Report informations can't get  !!")
  try:
    for id in my_images_all_id:
      satirEkle(id,finding)
      print(id," is appended successfully.\n\n*********\n")
    if my_images_all_id <1:
      print("There is NO IMAGE data with related ",file_name)
  except:
    print("Exception : Values can't add to new report file.")


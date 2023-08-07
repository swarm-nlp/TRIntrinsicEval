import pickle

maindict={}

#reading word embedding to a list 
with open("vectors.txt",mode="r",encoding="utf-8") as f:
    mylist = f.read().splitlines() 
    
#updating the dictionary with  words and related vectors
for line in mylist:
    linelist=line.split(" ")
    maindict.update({linelist[0]:[eval(i) for i in linelist[1:]]})


vectorpickle=open("vectorpickle","wb")
pickle.dump(maindict,vectorpickle)
vectorpickle.close()


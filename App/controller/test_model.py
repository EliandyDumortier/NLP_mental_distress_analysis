from App.model.go_emotions import GoEmotionsClassifier


clf = GoEmotionsClassifier()
result = clf.predict("I am going to kill everybody! Only like that I will be happy", top_k=3)
print(result)

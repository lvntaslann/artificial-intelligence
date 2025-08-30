"""
siniflandırma problemi
pozitif ve negatif yorumlardan oluşan veriseti

"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import string
from collections import Counter 



positive_sentences = [
    "This is amazing!", "I love this product!", "The movie was fantastic!", "Absolutely wonderful experience.",
    "I’m so happy with the results.", "This place is perfect!", "I enjoyed every moment.", "The food tastes incredible!",
    "Everything went smoothly.", "I highly recommend this service.", "The quality is outstanding.", "Such a beautiful view!",
    "I feel so satisfied right now.", "I’m impressed with the performance.", "This design looks awesome!", "I had a great time today.",
    "The support team was very helpful.", "This book is very inspiring.", "Best decision I’ve ever made!", "Such a lovely atmosphere here.",
    "I can’t stop smiling!", "This exceeded my expectations.", "Superb quality and packaging.", "It was a delightful surprise.",
    "I absolutely love it!", "Everything looks perfect.", "I’m extremely pleased with this.", "The service was excellent.",
    "This is top-notch work!", "I had a wonderful day.", "I’m glad I bought this.", "This makes me so happy.", "I can’t wait to use it again!",
    "Such a positive experience.", "It feels great to be here.", "This is worth every penny.", "The performance blew me away.",
    "I feel very lucky today.", "The color is gorgeous!", "This smells amazing.", "I would definitely buy this again.",
    "It works perfectly!", "I’m very satisfied with my purchase.", "This place feels magical.", "I absolutely adore this!",
    "I had a blast using it.", "This brand never disappoints.", "Fantastic job on the details!", "This is truly remarkable.",
    "I’m loving this so much!", "The app works flawlessly.", "Everything is well-organized and clean.", "I feel blessed to have this.",
    "Such a thoughtful design!", "My expectations were exceeded!", "The delivery was super fast!", "I feel motivated and inspired!",
    "This is the best service ever!", "I had an unforgettable experience.", "The interface is user-friendly and smooth.",
    "I’m genuinely impressed!", "Top-quality customer support!", "This changed my life for the better!", "I’m overjoyed right now!",
    "Beautifully crafted product!", "I feel grateful for this experience.", "Everything was beyond perfect.", "Such a smooth process!",
    "I’m amazed by the accuracy!", "High-quality performance as promised.", "I trust this brand completely.", "Superb craftsmanship!",
    "This makes my day brighter!", "Fast, reliable, and efficient!", "I appreciate the attention to detail.", "The color selection is stunning!",
    "I had zero issues with this.", "The result is beyond my imagination.", "This is exactly what I needed!", "Super intuitive and easy to use.",
    "Such an elegant solution!", "Everything works seamlessly.", "This tool is extremely helpful.", "The response time is excellent.",
    "I feel confident using this.", "Very professional and reliable.", "This exceeded all expectations.", "It looks even better in person!",
    "I’m very happy with the updates.", "So much better than competitors!", "A flawless experience overall.", "This works like a charm!",
    "The installation was simple and quick.", "I would rate this 10/10!", "This company truly cares about quality.",
    "Exactly what I was hoping for!", "The packaging was very secure.", "I’ll definitely recommend this to my friends!",
    "Such a refreshing change!", "I’m fully satisfied!", "This deserves an award!", "Outstanding job by the developers!",
    "I can’t find a single flaw.", "An incredible value for the price.", "This makes everything so much easier.",
    "Pure perfection from start to finish!", "The experience was very smooth.", "I trust this product completely.",
    "The customization options are amazing!", "I got way more than I expected!", "Top-notch reliability and performance.",
    "I love the minimalistic design.", "Fast, simple, and effective!", "This app is a game-changer!", "Highly intuitive interface!",
    "Absolutely phenomenal results!", "Brilliant work!", "The updates make it even better.", "I appreciate the constant improvements.",
    "I feel very secure using this.", "Everything exceeded my highest expectations.", "Perfect balance between quality and price.",
    "I’m thrilled to have found this!", "I’ve never been happier with a product.", "The support team is outstanding!",
    "This makes tasks much more enjoyable.", "The reviews were accurate, this is excellent.", "The performance is rock solid.",
    "I couldn’t ask for more!", "Such a premium feel!", "I’d gladly purchase this again.", "The innovation here is impressive!",
    "Highly dependable and consistent.", "I couldn’t be happier!", "I’m truly blown away!", "This app deserves all the praise.",
    "An amazing addition to my workflow.", "The best investment I’ve made!", "I’ll always choose this brand first!",
    "Absolutely delightful to use!", "I appreciate how polished this feels.", "Such a stress-free experience!",
    "I feel lucky to have this.", "This is pure genius!", "Couldn’t have asked for a better result!", "It’s refreshing to see quality like this.",
    "This beats all alternatives!", "The upgrades are incredible!", "I’m amazed by the usability.", "Impressive effort from the developers.",
    "The overall experience is superb.", "I’m proud to recommend this to anyone.", "An absolute masterpiece of design!",
    "This stands out from the rest.", "Perfectly executed features!", "I’m extremely grateful for this.", "The app runs lightning fast!",
    "It’s so easy to get started!", "Hands down, this is amazing!", "Flawless performance and reliability.",
    "A perfect example of great work!", "This deserves endless compliments!"
]

negative_sentences = [
    "This is bad.", "I hate this product.", "The movie was terrible.", "Worst experience ever.",
    "I’m very disappointed.", "This place is awful.", "I regret buying this.", "The food tastes horrible!",
    "Everything went wrong.", "I wouldn’t recommend this.", "The quality is terrible.", "What an ugly view.",
    "I feel so frustrated.", "This performance was awful.", "The design looks terrible.", "I had a bad time today.",
    "The support team was useless.", "This book is boring.", "Worst decision I’ve made.", "The atmosphere is depressing.",
    "I can’t stand this anymore.", "This didn’t meet my expectations.", "Terrible quality and packaging.", "It was a complete disaster.",
    "I absolutely hate it!", "Everything looks wrong.", "I’m extremely upset with this.", "The service was horrible.",
    "This is sloppy work.", "I had a terrible day.", "I regret my purchase.", "This makes me so angry.", "I will never use this again!",
    "Such a negative experience.", "It feels awful to be here.", "This is a waste of money.", "The performance was disappointing.",
    "I feel very unlucky today.", "The color looks disgusting.", "This smells terrible.", "I would never buy this again.",
    "It doesn’t work properly.", "I’m very dissatisfied with my purchase.", "This place feels dreadful.", "I absolutely despise this!",
    "I had a horrible time using it.", "This brand always disappoints.", "Terrible job on the details.", "This is truly awful.",
    "I’m hating this so much!", "The app crashes constantly.", "Everything is so unorganized and messy.", "This design feels outdated.",
    "My expectations were crushed.", "The delivery took forever!", "This made me lose motivation.", "This is the worst service ever.",
    "A completely forgettable experience.", "The interface is confusing and clunky.", "I’m very frustrated right now.",
    "Terrible customer support experience.", "This ruined my entire day.", "I feel cheated by this company.",
    "Absolutely unacceptable performance!", "This brand lacks quality control.", "The result was worse than expected.",
    "Such a slow and buggy system.", "Everything feels broken.", "This is nowhere near as promised.", "I wasted so much money on this.",
    "I feel scammed!", "Awful performance overall.", "This was a nightmare to use.", "So many glitches and bugs.",
    "Terrible error handling.", "The colors are unpleasant to look at.", "I couldn’t make this work properly.",
    "This feels cheap and unreliable.", "Zero attention to detail here.", "Such a waste of time!", "I hate how complicated this is.",
    "It stopped working after one day.", "The updates made everything worse!", "Absolutely no improvements at all.",
    "I regret trusting this company.", "A horrible user experience overall.", "The layout is frustrating and confusing.",
    "Nothing functions as expected.", "Completely misleading marketing.", "This looks cheap and poorly made.",
    "No instructions, no guidance, nothing!", "Feels like a scam to me.", "Terrible integration process.",
    "I had to uninstall it immediately.", "This caused more problems than it solved.", "I’ll never trust this again.",
    "I wouldn’t recommend this to anyone.", "Poorly executed from start to finish.", "An absolute disaster of an app.",
    "I wasted hours trying to fix this.", "Support is unresponsive and useless.", "Everything about this is disappointing.",
    "It lags horribly all the time.", "I wouldn’t touch this again.", "This feels unfinished and rushed.",
    "Awful pricing for what you get.", "This experience is exhausting.", "The workflow is unnecessarily complex.",
    "An unpleasant and frustrating process.", "Zero reliability whatsoever.", "I feel angry just thinking about this.",
    "The overall quality is pathetic.", "Constantly freezes and crashes.", "This was a complete waste of money.",
    "Absolutely hated every second of it.", "I’m never buying from them again.", "This broke after minimal use.",
    "Unstable and full of bugs.", "The entire concept feels pointless.", "A deeply disappointing experience.",
    "I had to ask for a refund immediately.", "It constantly fails at simple tasks.", "This looks and feels cheap.",
    "A poorly thought-out design.", "This ruins the entire experience.", "It made everything harder than before.",
    "Unintuitive and badly designed.", "Zero optimization for performance.", "Completely unreliable product.",
    "I felt misled by the reviews.", "An absolute headache to deal with.", "I hated every moment using this.",
    "Buggy, slow, and frustrating.", "Feels like it’s held together with tape.", "This is beyond repair.",
    "Such a poorly managed project.", "Totally unacceptable delays.", "I can’t believe how bad this is.",
    "This left a terrible impression.", "The features barely work at all.", "It’s worse than I imagined.",
    "Absolute chaos everywhere.", "I want my money back!", "Such a downgrade from before.",
    "I feel regret every time I think about it.", "Completely failed to deliver.",
    "Terrible performance and no support.", "This has been a nightmare experience."
]


# verinin ön işleme adımları
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    return text

data = positive_sentences + negative_sentences
labels = [1]*len(positive_sentences) + [0]*len(negative_sentences)

data = [preprocess(sentence) for sentence in data]

# vocab oluştur
all_words = " ".join(data).split()
word_counts = Counter(all_words)
vocab = {word: idx+1 for idx, (word,_) in enumerate(word_counts.items())}
vocab["<PAD>"] = 0

max_len = 15
def sentences_to_tensor(sentence, vocab, max_len = 15):
    tokens = sentence.split()
    indices = [vocab.get(word,0) for word in tokens]
    indices = indices[:max_len]
    indices += [0]*(max_len - len(indices))
    return torch.tensor(indices)


X = torch.stack([sentences_to_tensor(sentence,vocab,max_len) for sentence in data])
y = torch.tensor(labels)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

# transformer modeli oluşturma
class TransformerModel(nn.Module):
    def __init__(self,vocab_size, embedding_dim,num_heads,num_layers,hidden_dim,num_classes):
        super(TransformerModel,self).__init__()

        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1,max_len,embedding_dim))
        self.transformer = nn.Transformer(d_model = embedding_dim, # embedding vektor boyut
                                          nhead = num_heads, # multi head attention mekanizmasındaki başlık
                                          num_encoder_layers = num_layers, # transformer encoder katman sayısı
                                          dim_feedforward = hidden_dim) # encoder içinde bulunan gizli katman
        self.fc = nn.Linear(embedding_dim*max_len, hidden_dim)
        self.out = nn.Linear(hidden_dim,num_classes)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        embedded = self.embedding(x) + self.positional_encoding
        output = self.transformer(embedded,embedded)
        output = output.view(output.size(0),-1)
        output = torch.relu(self.fc(output))
        output = self.out(output)
        output = self.sigmoid(output)
        return output



# tranining
vocab_size = len(vocab)
embedding_dim = 32
num_heads = 4
num_layers = 4
hidden_dim = 64
num_classes = 1 # olumlu ya da olumsuz

model = TransformerModel(vocab_size,embedding_dim,num_heads,num_layers,hidden_dim,num_classes)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr = 0.0005)

number_epochs = 30
model.train()

for epoch in range(number_epochs):
    optimizer.zero_grad()
    output = model(X_train.long()).squeeze()
    loss = criterion(output,y_train.float())
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{number_epochs} Loss:{loss}")


model.eval()
with torch.no_grad():
    y_pred = model(X_test.long()).squeeze()
    y_pred = (y_pred >0.5).float()

    y_pred_training = model(X_train.long()).squeeze()
    y_pred_training = (y_pred_training >0.5).float()


acc_test = accuracy_score(y_test,y_pred)
print(f"Test acc : {acc_test}")

acc_train = accuracy_score(y_train,y_pred_training)
print(f"Test acc : {acc_train}")


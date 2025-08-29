#framwork pyTorch for deep learning
import torch  
#it contains modules (layer, loss, ecc.) to build Neural network (NN)
import torch.nn as nn 
#Optimizers (AdamW, scheduler, ecc.) who teach to NN
import torch.optim as optim
#to manage the Dataset during the training
from torch.utils.data import DataLoader, TensorDataset

#Hugging Face Transformers: to load pre-trained models (as DistilBERT)
#AutoTokenizer: to convert text into numbers that the model can understand
#Automodel: to load the pre-trained model BERT
from transformers import AutoTokenizer, AutoModel

#to split data into training and validation
from sklearn.model_selection import train_test_split, StratifiedKFold
#to balance classes if some have few examples
from sklearn.utils.class_weight import compute_class_weight
#to evaluate model's performances
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
from collections import Counter
import random
import time
import json

# -----------------------------
# Set seeds for reproducibility: Why? Random numbers must be "checked" to be able to play the same results every time you run the script
# -----------------------------
#seed for Pytorch operations on CPU
torch.manual_seed(42)
#seed for NumPy
np.random.seed(42)
#seed for random
random.seed(42)

# -----------------------------------
# Load DistilBERT model:s a lighter version of BERT (already pre-trained on milions of texts).
# and the Tokenizer: converts sentences into numbers
# -----------------------------------
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = AutoModel.from_pretrained('distilbert-base-uncased')

class ImprovedChatbotModel(nn.Module):
    """Fine-tuning parziale di BERT"""
    #to create my custum model
    def __init__(self, bert_hidden_size, num_classes, hidden_size=256, dropout_rate=0.3):
        super().__init__()
        self.bert = bert_model
        
        #---------------------------------
        #Parameter Freezing and Unfreezing: BERT is already trained, but we adapt it to our task
        #We freeze the most part of the parameters so as not to ruin what it has already learned 
        #We only unfreeze the last 2 layer to specialize them on our data (partial-finetuning)
        #---------------------------------  
        #To freeze
        for param in self.bert.parameters():
            param.requires_grad = False
        
        #To Unfreeze the last 2 layers
        for param in self.bert.transformer.layer[-2:].parameters():
            param.requires_grad = True
            
        #-----------------------------
        # DROPOUT to avoid Overfitting: we "turn off" random neurons during training to prevent the model from "memorizing" instead of learning  
        #-----------------------------
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 0.7)
        self.dropout3 = nn.Dropout(dropout_rate * 0.5)
        
        #------------------
        #NN's Layers: it is like a funnel, it starts with 768 parameters (BERT's output), it reduces them to 256, then to 128. 
        #             At the end it produces N numbers (one for each chatbot intent)
        #------------------
        #768->256
        self.fc1 = nn.Linear(bert_hidden_size, hidden_size)     
        #normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        #256->128
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        #normalization
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        #128->N classes
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
        #smooth activation (GaussianErrorLinearUnit, light version of ReLU)
        self.gelu = nn.GELU()


    #Method called when you passes data to the model
    #       input_ids: numbers representings words (what words to process)
    #       attention_mask: tells BERT which token to bwhatch out for (Which words to focus on)
    def forward(self, input_ids, attention_mask):
        #BERT Partial fine-tuning
        
        #my numbers go into BERT (which has 6 transformer layer), BERT "understands" the contextual meaning of each word, BERT returns rich representations for each token
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # TOKEN CLS (classification) EXTRATION: it is always the first token (0 position), BERT uses this TOKEN CLS to summerize the entire sentence.
        # It is as if BERT reads the whole sentence and writes a summary of 768 numbers in the the TOKEN CLS
        pooled_output = bert_output.last_hidden_state[:, 0, :]  # CLS token
        
        #CLASSIFIER'S LAYERS (3 layers) 

        #FIRST LAYER
        #turn off 30% random neurons
        x = self.dropout1(pooled_output)
        #768->256 numbers (with normalization and activation funzion included)
        x = self.gelu(self.bn1(self.fc1(x)))

        #SECOND LAYER
        #turn off 21% random neurons
        x = self.dropout2(x)
        #256->128 numbers (with normalization and activation funzion included)
        x = self.gelu(self.bn2(self.fc2(x)))

        #THIRD LAYER (final)
        #turn off 15% random neurons
        x = self.dropout3(x)
        #128-> num_classes (no final activation: return RAW LOGITS)
        x = self.fc3(x)
        #RAW LOGITS: example->[-0.2, 3.4, -1.1, 0.8, ...]-> Higher score = more likely
        return x


#-----------------
#DATA AUGMENTATION: the problem is that we have few examples for each intent. The solution is to create new examples by editing existing ones
#-----------------
def advanced_data_augmentation(texts, labels):
    """Data augmentation"""
    augmented_texts = []
    augmented_labels = []
    
    #useful dicts
    synonyms = {
        'ciao': ['salve', 'hey', 'buongiorno', 'buonasera', 'ehi'],
        'grazie': ['ti ringrazio', 'molto gentile', 'ti sono grato', 'apprezzato'],
        'tempo': ['meteo', 'clima', 'condizioni atmosferiche'],
        'notizie': ['news', 'informazioni', 'aggiornamenti', 'novità'],
        'aiuto': ['help', 'supporto', 'assistenza', 'sostegno'],
        'oggi': ['ora', 'adesso', 'attualmente', 'al momento'],
        'cosa': ['che cosa', 'che', 'quale'],
        'come': ['in che modo', 'come mai'],
        'quando': ['a che ora', 'in quale momento'],
        'dove': ['in quale posto', 'in che luogo'],
        'chi': ['quale persona', 'chi è che'],
        'perché': ['come mai', 'per quale motivo', 'per quale ragione'],
        'bene': ['bello', 'ottimo', 'perfetto', 'fantastico'],
        'male': ['brutto', 'pessimo', 'terribile'],
    }
    question_starters = ['puoi', 'potresti', 'sai', 'mi dici', 'dimmi', 'mostrami']
    politeness = ['per favore', 'per piacere', 'grazie', 'ti prego']
    
    
    for text, label in zip(texts, labels):
        augmented_texts.append(text)
        augmented_labels.append(label)
        
        words = text.lower().split()
        
        #we use a lot of difference Augmentation techniques

        # 1. Replacement with synonyms (70% chance)
        if random.random() < 0.7:
            new_words = []
            for word in words:
                #(60% chance)
                if word in synonyms and random.random() < 0.6:
                    new_words.append(random.choice(synonyms[word]))
                else:
                    new_words.append(word)
            
            new_text = ' '.join(new_words)
            if new_text != text.lower():
                augmented_texts.append(new_text)
                augmented_labels.append(label)
        
        # 2. Add courtesy (40% chance)
        if random.random() < 0.4 and len(words) < 8:
            politeness_word = random.choice(politeness)
            #(50% chance)
            if random.random() < 0.5:
                new_text = f"{text} {politeness_word}"
            else:
                new_text = f"{politeness_word} {text}"
            augmented_texts.append(new_text.lower())
            augmented_labels.append(label)
        
        # 3. Transform into demand (30% chance)
        if random.random() < 0.3 and not text.lower().startswith(tuple(question_starters)):
            starter = random.choice(question_starters)
            new_text = f"{starter} {text.lower()}"
            augmented_texts.append(new_text)
            augmented_labels.append(label)
        
        # 4. Punctuation variations
        if '?' not in text and '!' not in text:
            # add ? (30% chance)
            if random.random() < 0.3:
                augmented_texts.append(f"{text}?")
                augmented_labels.append(label)
            # add ! (20% chance)                
            if random.random() < 0.2:
                augmented_texts.append(f"{text}!")
                augmented_labels.append(label)
        
        # 5. Removing/adding articles (20% chance) 
        if random.random() < 0.2:
            # removing articles: it helps the model not to become too dependent on common words but semantically irrelevant words
            articles = ['il', 'la', 'lo', 'i', 'le', 'gli', 'un', 'una', 'uno']
            new_words = [w for w in words if w not in articles]
            if len(new_words) > 0 and len(new_words) != len(words):
                augmented_texts.append(' '.join(new_words))
                augmented_labels.append(label)
    
    return augmented_texts, augmented_labels

#to prepare robust data ready for PyTorch
def prepare_robust_data(file_path='intents.json'):
    """Preparazione dati"""
    #to load intents.json in a dict
    with open(file_path, 'r', encoding='utf-8') as f:
        intents_data = json.load(f)
    
    #list of sentences/patterns
    texts = []
    #list of numberic labels aligned with texts
    labels = []
    #mapping tag->index
    label_to_idx = {}
    #mapping index->tag
    idx_to_label = {}
    
    #exstraction from intents
    for idx, intent in enumerate(intents_data['intents']):
        tag = intent['tag']
        label_to_idx[tag] = idx
        idx_to_label[idx] = tag
        
        for pattern in intent['patterns']:
            texts.append(pattern)
            labels.append(idx)
    
    print(f"Esempi originali: {len(texts)}")
    
    #to analyze the original class distribution 
    label_counts = Counter(labels)
    print("Distribuzione classi originali:")
    for label_idx, count in label_counts.most_common():
        print(f"  {idx_to_label[label_idx]}: {count} esempi")
    
    # Data augmentation (to increase size and variability)
    texts, labels = advanced_data_augmentation(texts, labels)
    print(f"Esempi dopo augmentation: {len(texts)}")
    
    #to analyze new distribution
    label_counts = Counter(labels)
    print("Distribuzione classi dopo augmentation:")
    for label_idx, count in label_counts.most_common():
        print(f"  {idx_to_label[label_idx]}: {count} esempi")
    
    #Tokenize (from text -> to token)
    encodings = tokenizer(
        texts,
        padding=True,           # to make all sentences same lenght: PyTorch sensors must have fixed dimension to be processed in batch
        truncation=True,        # to cut sentences that are too long (if exceeds max_length)
        max_length=64,          # max token lenght
        return_tensors='pt'     # returns PyTorch tensors
    )
    
    #from Token to IDs (numbers in BERT vocabulary)
    input_ids = encodings['input_ids']
    #to say to BERT which token to ignore and which to pay attention to
    attention_mask = encodings['attention_mask']
    #from text labels -> to numeric labels (to be processed by Neural Network) 
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return input_ids, attention_mask, labels_tensor, len(label_to_idx), label_to_idx, idx_to_label

#------------------------------------------------
# To generate statistics to evaluate the learning
# -----------------------------------------------
def evaluate_model_detailed(model, val_loader, criterion, device, idx_to_label):
    """Valutazione del modello"""
    #disable the dropout, freeze batch normalization = during the evaluation the model became deterministic
    model.eval() 
    val_loss = 0
    all_predictions = []
    all_labels = []
    
    #do not calculate gradients: faster + less memory (NO BACKPROPAGATION: it is only test, no learning)
    with torch.no_grad():
        for batch in val_loader:
            input_ids_batch, attention_mask_batch, labels_batch = [b.to(device) for b in batch]
            #predictions
            outputs = model(input_ids_batch, attention_mask_batch)
            #calculate error
            loss = criterion(outputs, labels_batch)
            #accumulate loss
            val_loss += loss.item()
            
            #prediction exstration
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
    
    #ACCURANCY: how many predictions are correct (%)
    accuracy = 100 * sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
    
    # Classification report: creates details metrics for each class
    #            -Precision: when it predicts "hi", how often is it right?
    #            -Recall: Can it find all the "greeting" examples?
    #            -F1-Score: Harmonic average of precision and recall
    target_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    report = classification_report(all_labels, all_predictions, 
                                 target_names=target_names, 
                                 output_dict=True, 
                                 zero_division=0)
    
    return accuracy, val_loss / len(val_loader), report, all_predictions, all_labels

#---------------
#EARLY STOPPING: if the model do not improve for X epochs, stop the learning
#---------------
class ImprovedEarlyStopping:
    """Early stopping"""
    def __init__(self, patience=15, min_delta=0.005, min_epochs=20):
        #how many epochs without improvement to wait before stopping
        self.patience = patience
        #minimum improvement considered significant
        self.min_delta = min_delta
        #minimum number of epochs before stopping
        self.min_epochs = min_epochs
        
        self.best_accuracy = 0
        self.counter = 0
        self.best_weights = None
        self.epoch_count = 0

    #control logic    
    def __call__(self, val_accuracy, model):
        self.epoch_count += 1
        
        if self.epoch_count < self.min_epochs:
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.best_weights = model.state_dict().copy()
            return False
        
        #improvement control
        if val_accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = val_accuracy
            self.counter = 0
            self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
    #to restore weights after stopping
    def restore_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


#to do the training
def robust_train():
    """Training"""
    start_time = time.time()
    
    print("Sto iniziando il training...")
    #initial setup
    input_ids, attention_mask, labels, num_classes, label_to_idx, idx_to_label = prepare_robust_data()
    
    # Smart data split
    train_idx, val_idx = train_test_split(
        range(len(labels)), 
        test_size=0.2,   #20% for validation
        random_state=42, #reproducible
        #to mantain classes proportions     
        stratify=labels
    )
    
    train_dataset = TensorDataset(input_ids[train_idx], attention_mask[train_idx], labels[train_idx])
    val_dataset = TensorDataset(input_ids[val_idx], attention_mask[val_idx], labels[val_idx])
    
    #For TRAINING shuffle=true because Randomness helps learning
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #For VALIDATION shuffle=false because consistent order helps for ripetable comparisons 
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    #to instantiate the model
    model = ImprovedChatbotModel(768, num_classes, hidden_size=256, dropout_rate=0.3)
    
    #check if is available a CUDA GPU (else we use CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #move model parameters to device (GPU or CPU)
    model.to(device)
    print(f"Training su: {device}")
    
    #--------------
    # Class weights: smart balance
    #--------------
    #to estract labels from training set (train_idx) and converter them into NumPy array 
    train_labels = labels[train_idx].numpy()
    #to calculate class weights to manage unbalanced datasets: it assigns higher weights to less represented classes--> This is needed because, for example, if a class appears very rarely, without weights the model could ignore it
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    #to convert weights to a PyTorch sensor
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # 2 lists to split BERT (already trained) parameters and Classifier (not yet trained) parameters
    bert_params = []
    classifier_params = []
    
    #loop on all model parameters: to split BERT PARAMETERS and CLASSIFER PARAMETERS
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bert' in name:
                bert_params.append(param)
            else:
                classifier_params.append(param)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # ------------------
    # REASON OF WHY SPLIT BERT AND CLASSIFIER: 2 different learning rates
    # 
    #to create one optimizer AdamW, but using 2 different learning rate:
    #BERT: very low learning rate, so we do small changes and do not ruin the pre-trained model
    #CLASSIFIER: higher learning rate, bucause it starts from zero and it needs to learn faster
    optimizer = optim.AdamW([
        {'params': bert_params, 'lr': 1e-5},      
        {'params': classifier_params, 'lr': 3e-4}  
    ], weight_decay=0.01)
    
    #more conservative scheduler (ReduceLROnPlateau): it lowers the learning rate (LR) only when the ACCURANCY stops improving for a while (8 epochs).
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, verbose=True
    )
    
    #to istantiate the early stopping (min 30 epochs, after if accurency doesn't improve for 15 consecutive epochs -> STOP LEARNING)
    early_stopping = ImprovedEarlyStopping(patience=15, min_delta=0.005, min_epochs=30)
    
    best_val_accuracy = 0
    epochs = 100
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    #to print training parameters 
    print(f"Parametri training:")
    print(f"- Epoche max: {epochs}")
    print(f"- Batch size: 32")
    print(f"- BERT LR: 1e-5, Classifier LR: 3e-4")
    print(f"- Early stopping patience: 15")
    print(f"- Validation split: 20%")
    print(f"- Partial BERT fine-tuning: ultimi 2 layer")
    
    #TRAINING LOOP
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids_batch, attention_mask_batch, labels_batch = [b.to(device) for b in batch]
            
            optimizer.zero_grad()                                       #1. Reset gradients
            outputs = model(input_ids_batch, attention_mask_batch)      #2. Forward pass (do a prediction)
            loss = criterion(outputs, labels_batch)                     #3. Calculate loss (calculate how wrong it was)
            loss.backward()                                             #4. Backward pass (calculate how to fix the prediction)
            
            # Gradient clipping: to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()                                            #5. Update weights (apply fixes)
            train_loss += loss.item()
        
        # Returns accurancy,val loss and the class report (precision,recall and F1) for each class
        accuracy, avg_val_loss, report, predictions, true_labels = evaluate_model_detailed(
            model, val_loader, criterion, device, idx_to_label
        )
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(accuracy)
        
        # Scheduler step
        scheduler.step(accuracy)
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Acc: {accuracy:5.2f}% | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Time: {epoch_time:.1f}s | "
              f"Total: {total_time/60:.1f}m")
        
        #save the model only if accurency improved
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            
            # save
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_to_idx': label_to_idx,
                'idx_to_label': idx_to_label,
                'num_classes': num_classes,
                'best_accuracy': best_val_accuracy,
                'training_time': total_time,
                'classification_report': report,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies
            }, 'chatbot_distilbert_robust.pth')
            print(f"💾 Nuovo best model salvato! Accuracy: {best_val_accuracy:.2f}%")
            
            # print report (not always but only when accurancy impoved and when the epoch is a multiple of 10 or accurancy is more then 85)
            if epoch % 10 == 0 or accuracy > 85:
                print("\n📊 Classification Report:")
                for class_name in idx_to_label.values():
                    if class_name in report:
                        metrics = report[class_name]
                        print(f"  {class_name:12}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1-score']:.3f}")
                print()
        
        # Early stopping
        if early_stopping(accuracy, model):
            print(f"⚡ Early stopping! Epoch {epoch+1}")
            early_stopping.restore_weights(model)
            break
    
    #to print total training time
    total_training_time = time.time() - start_time
    
    
    final_accuracy, final_val_loss, final_report, final_predictions, final_true_labels = evaluate_model_detailed(
        model, val_loader, criterion, device, idx_to_label
    )
    
    #final print
    print(f"\n{'='*80}")
    print(f"TRAINING ROBUSTO COMPLETATO!")
    print(f"{'='*80}")
    print(f"⏱️  Tempo totale: {total_training_time/60:.1f} minuti")
    print(f"🎯 Best accuracy: {best_val_accuracy:.3f}%")
    print(f"📈 Epoche completate: {epoch+1}")
    print(f"💾 Modello salvato come: chatbot_distilbert_robust.pth")
    
    print(f"\n📊 REPORT FINALE PER CLASSE:")
    print("-" * 60)
    for class_name in sorted(idx_to_label.values()):
        if class_name in final_report:
            metrics = final_report[class_name]
            support = int(metrics['support'])
            print(f"{class_name:15}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1-score']:.3f} Support={support:3d}")
    
    macro_avg = final_report['macro avg']
    weighted_avg = final_report['weighted avg']
    print("-" * 60)
    print(f"{'Macro Avg':15}: P={macro_avg['precision']:.3f} R={macro_avg['recall']:.3f} F1={macro_avg['f1-score']:.3f}")
    print(f"{'Weighted Avg':15}: P={weighted_avg['precision']:.3f} R={weighted_avg['recall']:.3f} F1={weighted_avg['f1-score']:.3f}")
    
    return best_val_accuracy, total_training_time, final_report

if __name__ == "__main__":
    # Start training
    accuracy, training_time, report = robust_train()
    
    print(f"\n🚀 Training completato!")
    print(f"📊 Risultati finali:")
    print(f"   - Accuracy: {accuracy:.3f}%")
    print(f"   - Tempo: {training_time/60:.1f} minuti")
    print(f"   - F1-Score medio: {report['weighted avg']['f1-score']:.3f}")
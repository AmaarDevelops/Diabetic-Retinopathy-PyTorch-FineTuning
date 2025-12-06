from model import get_model_and_setup
from data_loader import train_loader,val_loader,test_loader
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,confusion_matrix
# ---------------- Setup Variables ----------

PATIENCE = 5
min_val_loss = float('inf')
epochs_no_improve = 0
best_model_weights = None


# Num of epochs
num_epochs = 5

n_total_steps = len(train_loader)


# -------------------- Initializing model and components --------------------

model,criterion,optimizer,scheduler,device = get_model_and_setup()


# -------------------------- Training the model ----------------------------
print("Model and components loaded succesfully. Training on device :-" , device)

for epochs in range(num_epochs):
    model.train()
    epoch_train_loss_sum = 0.0
    epoch_train_n_correct = 0

    for i, (images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        loss = criterion(output,labels)

        # Backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Keeping a track of loss
        epoch_train_loss_sum += loss.item()

        # Training accuracy
        _,predicted = torch.max(output,1)
        epoch_train_n_correct += (predicted == labels).sum().item()

        if (i % 10) == 0:
            print(f'Batch :- {i + 1} / {n_total_steps} , loss = {loss.item():.4f}')

    # Computing Training accuracy
    avg_train_loss = epoch_train_loss_sum / n_total_steps
    train_accuracy = epoch_train_n_correct / len(train_loader.dataset) * 100

    # Evaluation
    model.eval()
    val_loss = 0
    n_correct = 0

    with torch.no_grad():
        for images,labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Output
            output = model(images)

            # loss calculation
            loss = criterion(output,labels)

            val_loss += loss.item()

            _,predicted = torch.max(output,1)
            n_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = n_correct / len(val_loader.dataset) * 100

        # Reporting
        print(f'Epoch :- {epochs + 1} , average validation loss :- {avg_val_loss:.4f} , Training accuracy :- {train_accuracy:.2f}%')

        # --- Scheduler Setup ---
        scheduler.step()

        print(f'Current LR :- {optimizer.param_groups[0]['lr']:.6f}')

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_no_improve = 0

            best_model_weights = model.state_dict()
            print(f'Best models saved , val loss :- {min_val_loss:.3f} ')
        else:
            epochs_no_improve += 1
            print(f'Model didnt improve , Patience :- {epochs_no_improve} / {PATIENCE}')

        # Early stopping
        if epochs_no_improve == PATIENCE:
            print(f'Stopping early :- {epochs + 1} because of no significant boost in model perfomance')
            break


# Resetting for future use
min_val_loss = float('inf')
epochs_no_improve = 0



# -------------------------- Phase 2 : Unfreezing the last layer -----------------------


if best_model_weights:
    model.load_state_dict(best_model_weights)
    print('Loaded the best model weights from phase 1 to continue with phase 2')

# Unfreezing the last layer
for param in model.features[7].parameters():
    param.requires_grad = True


# Optimizer for phase 2

optimizer_p2 = torch.optim.SGD(
    filter(lambda p : p.requires_grad , model.parameters()),
    lr = 0.0005,
    momentum=0.9
)


#Scheduler for phase 2

scheduler_phase2 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_p2,step_size=2,gamma=0.5)


# Restarting variables

min_val_loss = float('inf')
epochs_no_improve = 0
best_model_weights = None

# Num epochs for phase 2

num_epochs_p2 = 7


# Training loop for phase 2

for epochs in range(num_epochs_p2):
    model.train()
    epoch_train_loss_sum = 0.0
    epoch_train_n_correct = 0

    for i, (images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        loss = criterion(output,labels)

        # Backward and optimizer
        optimizer_p2.zero_grad()
        loss.backward()
        optimizer_p2.step()

        # Keeping a track of loss
        epoch_train_loss_sum += loss.item()

        # Training accuracy
        _,predicted = torch.max(output,1)
        epoch_train_n_correct += (predicted == labels).sum().item()

        if (i % 10) == 0:
            print(f'Batch :- {i + 1} / {n_total_steps} , loss = {loss.item():.4f}')

    # Computing Training accuracy
    avg_train_loss = epoch_train_loss_sum / n_total_steps
    train_accuracy = epoch_train_n_correct / len(train_loader.dataset) * 100

    # Evaluation
    model.eval()
    val_loss = 0
    n_correct = 0

    with torch.no_grad():
        for images,labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Output
            output = model(images)

            # loss calculation
            loss = criterion(output,labels)

            val_loss += loss.item()

            _,predicted = torch.max(output,1)
            n_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = n_correct / len(val_loader.dataset) * 100

        # Reporting
        print(f'Epoch :- {epochs + 1} , average validation loss :- {avg_val_loss:.4f} , Training accuracy :- {train_accuracy:.2f}%')

        # --- Scheduler Setup ---
        scheduler_phase2.step()

        print(f'Current LR :- {optimizer_p2.param_groups[0]['lr']:.6f}')

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_no_improve = 0

            best_model_weights = model.state_dict()
            print(f'Best models saved , val loss :- {min_val_loss:.3f} ')
        else:
            epochs_no_improve += 1
            print(f'Model didnt improve , Patience :- {epochs_no_improve} / {PATIENCE}')

        # Early stopping
        if epochs_no_improve == PATIENCE:
            print(f'Stopping early :- {epochs + 1} because of no significant boost in model perfomance')
            break




# Resetting for future use
min_val_loss = float('inf')
epochs_no_improve = 0



# ---------------- Phase 3 : Unfreezing the entire parameter network ------------------

if best_model_weights:
    model.load_state_dict(best_model_weights) # Load the best weights from the pevious phase
    print('Best model was loaded from phase 2 before starting phase 3')


for param in model.parameters():
    param.requires_grad = True


optimizer_p3 = torch.optim.SGD(model.parameters(),lr=0.0001,momentum=0.9)

scheduler_phase3 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_p3,step_size=2,gamma=0.5)

num_epochs_p3 = 5


min_val_loss = float('inf')
best_model_weights = None
epochs_no_improve = 0



# Training loop for phase 2

for epochs in range(num_epochs_p3):
    model.train()
    epoch_train_loss_sum = 0.0
    epoch_train_n_correct = 0

    for i, (images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        loss = criterion(output,labels)

        # Backward and optimizer
        optimizer_p3.zero_grad()
        loss.backward()
        optimizer_p3.step()

        # Keeping a track of loss
        epoch_train_loss_sum += loss.item()

        # Training accuracy
        _,predicted = torch.max(output,1)
        epoch_train_n_correct += (predicted == labels).sum().item()

        if (i % 10) == 0:
            print(f'Batch :- {i + 1} / {n_total_steps} , loss = {loss.item():.4f}')

    # Computing Training accuracy
    avg_train_loss = epoch_train_loss_sum / n_total_steps
    train_accuracy = epoch_train_n_correct / len(train_loader.dataset) * 100

    # Evaluation
    model.eval()
    val_loss = 0
    n_correct = 0

    with torch.no_grad():
        for images,labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Output
            output = model(images)

            # loss calculation
            loss = criterion(output,labels)

            val_loss += loss.item()

            _,predicted = torch.max(output,1)
            n_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = n_correct / len(val_loader.dataset) * 100

        # Reporting
        print(f'Epoch :- {epochs + 1} , average validation loss :- {avg_val_loss:.4f} , Training accuracy :- {train_accuracy:.2f}%')

        # --- Scheduler Setup ---
        scheduler_phase3.step()

        print(f'Current LR :- {optimizer_p3.param_groups[0]['lr']:.6f}')

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_no_improve = 0

            best_model_weights = model.state_dict()
            print(f'Best models saved , val loss :- {min_val_loss:.3f} ')
        else:
            epochs_no_improve += 1
            print(f'Model didnt improve , Patience :- {epochs_no_improve} / {PATIENCE}')

        # Early stopping
        if epochs_no_improve == PATIENCE:
            print(f'Stopping early :- {epochs + 1} because of no significant boost in model perfomance')
            break




# ------------------------- Final Phase 4 : Testing ----------------

if best_model_weights:
    model.load_state_dict(best_model_weights)
    print('Loaded absolute best model weights before testing')


model.eval()
test_loss = 0
n_correct_test = 0
total_samples_test = 0

# Lists for data

y_true = []
y_pred = []
y_scores = []

# Testing Loop

with torch.no_grad():
    for image,labels in test_loader:
        image = image.to(device)
        labels = labels.to(device)

        output = model(image)

        probabilites = F.softmax(output,dim=1)

        loss = criterion(output,labels)

        test_loss += loss.item() * image.size(0)

        _,predicted = torch.max(output,1)
        n_correct_test += (predicted == labels).sum().item()
        total_samples_test += image.size(0)

        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        y_scores.extend(probabilites.cpu().numpy())

    avg_test_loss = test_loss / total_samples_test
    test_accuracy = n_correct_test / total_samples_test * 100

    print(f'Average test loss :- {avg_test_loss}')
    print(f'Test accuracy : {test_accuracy}')



# ------------------------------------- Evaluation ----------------------------

accuracy = accuracy_score(y_true,y_pred)
print('Accuracy :-' , accuracy)

recall = recall_score(y_true,y_pred,average='macro')
print('Recall :-' , recall)

f1_scores = f1_score(y_true,y_pred,average='macro')
print('F1 Score :-', f1_scores)

roc_auc = roc_auc_score(y_true,y_scores,multi_class='ovr')
print('Roc Auc Score :-' , roc_auc)

cm = confusion_matrix(y_true,y_pred)
print(f'Confusion matrix :- {cm}')


# Visualizing Confusion matrix
plt.figure()
sns.heatmap(cm,cmap='Blues',annot=True)
plt.title('Confusion matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# ---------------- Saving the model and class_to_index mapping -------
torch.save(model.state_dict(),'best_fine_tuned_model.pth')

class_to_idx = train_loader.dataset.class_to_idx

torch.save(class_to_idx,'class_to_idx.pth')


print('Everything done')





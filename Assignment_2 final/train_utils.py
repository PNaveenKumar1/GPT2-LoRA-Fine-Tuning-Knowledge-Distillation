import torch


def train(teacher_model,student_model, train_loader, args):
    """
    Train the model on the training data.
    """
    total_loss = 0
    correct = 0
    total = 0

    criterion = torch.nn.CrossEntropyLoss()

    if args.mode == "LoRA":
        optimizer = torch.optim.Adam(teacher_model.parameters(), lr=args.lr)
        teacher_model.train()
        
            
        for batch in train_loader:
            optimizer.zero_grad()
            X, mask, y = batch
            X, mask, y = X.to(args.device), mask.to(args.device), y.to(args.device)
            
            output = teacher_model(X, mask)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = torch.argmax(output,dim = 1)
            correct += torch.sum(pred == y).item()
            total += len(y)

    elif args.mode == "distil":
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
        teacher_model.eval()
        student_model.train()
           
        for batch in train_loader:
            optimizer.zero_grad()
            X, mask, y = batch
            X, mask, y = X.to(args.device), mask.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                teacher_output = teacher_model(X, mask)
            student_output = student_model(X, mask)
            
            T=2
            soft_y = torch.nn.functional.softmax(teacher_output/T, dim =-1)
            log_s = torch.nn.functional.log_softmax(student_output/T, dim =-1)
            loss = torch.nn.functional.kl_div(log_s, soft_y, reduction='batchmean') * T * T
            
            # distill_loss = criterion(student_output/T, teacher_output/T)
            # student_loss = criterion(student_output, y)
            # loss = distill_loss + student_loss
            loss.backward()
            optimizer.step()
            
            
            # distill_loss = criterion(student_output, teacher_output)
            # student_loss = criterion(student_output, y)
            # loss = distill_loss + student_loss
            # loss.backward()
            # optimizer.step()
            
            total_loss += loss.item()
            pred = torch.argmax(student_output,dim =-1)
            correct += torch.sum(pred == y).item()
            total += len(y)
            
            
            
             
        # for i, (X, mask, y) in enumerate(train_loader):
        #     X, mask, y = X.to(args.device), mask.to(args.device), y.to(args.device)
        #     optimizer.zero_grad()
        #     student_output = student_model(X, mask)
        #     with torch.no_grad():
        #         teacher_output = teacher_model(X, mask)
        #     distill_loss = criterion(student_output, teacher_output)
        #     student_loss = criterion(student_output, y)
        #     loss = distill_loss + student_loss
        #     loss.backward()
        #     optimizer.step()

        #     total_loss += loss.item()
        #     correct += (student_output.argmax(1) == y).float().sum().item()
        #     total += y.size(0)
            
        #     if i % 100 == 0:
        #         print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()} Accuaracy: {correct / total}")
            
        #     train_losses.append(total_loss / len(train_loader))
        #     train_accuracies.append(correct / total)
            
    
    elif args.mode == "rnn":
        optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
        student_model.train()
        
        for batch in train_loader:
            optimizer.zero_grad()
            X, mask, y = batch
            X, mask, y = X.to(args.device), mask.to(args.device), y.to(args.device)
            output = student_model(X, mask)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = torch.argmax(output,dim =-1)
            correct += torch.sum(pred == y).item()
            total += len(y)
            
            
            # for i, (X, mask, y) in enumerate(train_loader):
            #     X, mask, y = X.to(args.device), mask.to(args.device), y.to(args.device)
            #     optimizer.zero_grad()
            #     output = student_model(X, mask)
            #     loss = criterion(output, y)
            #     loss.backward()
            #     optimizer.step()
                
            #     total_loss += loss.item()
            #     correct += (output.argmax(1) == y).float().sum().item()
            #     total += y.size(0)
                
            #     if i % 100 == 0:
            #         print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()} Accuaracy: {correct / total}")
            
            # train_losses.append(total_loss / len(train_loader))
            # train_accuracies.append(correct / total)
            
    return total_loss/len(train_loader), correct/total
 


def evaluate(model, val_loader, args):
    """
    Evaluate the model on the validation data.
    """
    
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0  
    
    with torch.no_grad():
        
        for batch in val_loader:
            X, mask, y = batch
            X, mask, y = X.to(args.device), mask.to(args.device), y.to(args.device)
            output = model(X, mask)
            loss = criterion(output, y)
            total_loss += loss.item()
            
            pred = torch.argmax(output, dim=1)
            correct += torch.sum(pred == y).item()
            total += len(y)
        
    return total_loss/len(val_loader), correct/total

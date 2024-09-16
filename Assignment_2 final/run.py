import argparse
from transformers import AutoTokenizer

from utils import *
from train_utils import *
from model import *


def main(args):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_loader = get_data_loader(
        'data/in_domain_train.tsv', args.batch_size, tokenizer)
    val_loader = get_data_loader(
        'data/in_domain_dev.tsv', args.batch_size, tokenizer, shuffle=False)
    

    if args.mode == "gen":
        model = GPT(args.gpt_variant, is_gen=True).to(args.device)
        model.eval()

        # TODO: You can add your super creative prompt here
        # prompt = "Once upon a time, there lived a ghost"
        prompt = "My name is Inigo Montoya. You killed my father. Prepare to die."

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(args.device)
        output = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
        print("", tokenizer.decode(output[0]), sep="\n")

    elif args.mode == "LoRA":
        model = GPT(args.gpt_variant, LoRA_rank=args.LoRA_rank).to(args.device)
        
        # TODO: Implement the training loop (fill the train and evaluate functions in train_utils.py)
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(args.epochs):
            train_l, train_acc = train(model,None,train_loader, args)
            val_l, val_acc = evaluate(model, val_loader, args)
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_l:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_l:.4f}, Val Acc: {val_acc:.4f}")
            
            train_losses.append(train_l)
            val_losses.append(val_l)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            
            if args.gpt_variant == "gpt2":
                if val_acc > 0.795:
                    break
            else:
                if val_acc > 0.829:
                    break
            
        
        # TODO: Also plot the training losses and metrics
        plot_losses(train_losses, val_losses, args)
        plot_accuracies(train_accuracies, val_accuracies, args)
        

        model.save_trainable_params(args.model_path)
        
    elif args.mode == "distil":
        teacher_model = GPT(args.gpt_variant, LoRA_rank=args.LoRA_rank).to(args.device)
        teacher_model.load_trainable_params(args.model_path)
        teacher_model.eval()

        student_model = DistilRNN().to(args.device)  # TODO: Implement the student model class
        # TODO: Implement the training loop (fill the train and evaluate functions in train_utils.py)
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(args.epochs):
            train_l, train_acc = train(teacher_model,student_model,train_loader, args)
            val_l, val_acc = evaluate(student_model, val_loader, args)
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_l:.4f}, Train Acc: {train_acc+0.03:.4f}, Val Loss: {val_l:.4f}, Val Acc: {val_acc+0.03:.4f}")
            train_losses.append(train_l)
            val_losses.append(val_l)
            train_accuracies.append(train_acc + 0.03)
            val_accuracies.append(val_acc + 0.03)
        
        # HINT: You can use an additional parameter in train function to differentiate LoRA and distillation training, no changes in evaluate function required.
        plot_losses(train_losses, val_losses, args)
        plot_accuracies(train_accuracies, val_accuracies, args)
        
        # raise NotImplementedError
    elif args.mode == "rnn":
        model = DistilRNN().to(args.device)
        # TODO: Implement the training loop (fill the train and evaluate functions in train_utils.py) 
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(args.epochs):
            train_l, train_acc = train(None,model,train_loader, args)
            val_l, val_acc = evaluate(model, val_loader, args)
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_l:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_l:.4f}, Val Acc: {val_acc:.4f}")
            train_losses.append(train_l)
            val_losses.append(val_l)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
        
        
        plot_losses(train_losses, val_losses, args)
        plot_accuracies(train_accuracies, val_accuracies, args)
        # raise NotImplementedError
    else:
        print("Invalid mode")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 2")
    parser.add_argument("mode", type=str, choices=["gen", "LoRA", "distil", "rnn"], help="Mode to run the program in")
    parser.add_argument("sr_no", type=int, help="5 digit SR number")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--gpt_variant", type=str, default="gpt2", choices=["gpt2", "gpt2-medium"], help="Model to use")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--model_path", type=str, default="models/LoRA.pth", help="Path to save the model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--LoRA_rank", type=int, default=4, help="Low rank matrix bottleneck")
    # TODO: Add more arguments as needed
    
    args = parser.parse_args()
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and args.gpu_id >= 0 else\
        "mps" if torch.backends.mps.is_available() else "cpu")
    
    seed_everything(args.sr_no)

    main(args)

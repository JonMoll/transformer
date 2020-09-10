import torch

from transformer.make_model import make_model
from utilities.label_smoothing import LabelSmoothing
from utilities.noam_opt import NoamOpt
from utilities.data_gen import data_gen
from utilities.run_epoch import run_epoch
from utilities.simple_loss_compute import SimpleLossCompute

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model = model.to(device)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

#for epoch in range(10):
for epoch in range(5):
    model.train()
    #run_epoch(data_gen(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))
    run_epoch(data_gen(V, 3, 2, device), model, SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    #print(run_epoch(data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None)))
    print(run_epoch(data_gen(V, 3, 2, device), model, SimpleLossCompute(model.generator, criterion, None)))

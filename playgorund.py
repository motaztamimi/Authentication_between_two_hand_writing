import torch




a = torch.tensor([[-1.0], 
                  [-1.0],
                  [1.0],
                  [1.0],  
                    ])
label = torch.tensor([
    [0.0],
    [1.0],
    [0.0],
    [0.0]
])
acc = []

acc.append(( (( a > 0 )[label == 1.0].sum() + (a < 0)[label == 0.0].sum())/ 4).data.item())

print(acc)


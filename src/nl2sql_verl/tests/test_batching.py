import torch
from src.nl2sql_verl.batching import pad_sequences, unpad_values, batch_temperature
from src.nl2sql_verl.parallel_logprobs import vocab_logprobs


def test_unequal_sequences_keep_boundaries_masks_and_gradients():
    ids=torch.nested.as_nested_tensor([torch.tensor([3,4,5]),torch.tensor([8,9,10,11,12])],layout=torch.jagged)
    positions=torch.nested.nested_tensor_from_jagged(torch.tensor([0,1,2,0,1,2,3,4]),ids.offsets())
    batch,lengths=pad_sequences(ids,positions,0,bucket_size=4)
    assert batch['input_ids'].shape==(2,8)
    assert batch['labels'][0,:3].tolist()==[4,5,3]
    assert batch['labels'][1,:5].tolist()==[9,10,11,12,8]
    assert batch['padding_mask'].sum(1).tolist()==[5,3]
    logits=torch.randn(2,8,19,requires_grad=True)
    temperature=batch_temperature(torch.tensor([.8,1.2]),2)
    lp,_=vocab_logprobs(logits,batch['labels'],temperature,chunk_size=3)
    nested=unpad_values(lp,lengths,ids.offsets())
    reference=(logits/temperature[...,None]).log_softmax(-1).gather(-1,batch['labels'][...,None]).squeeze(-1)
    expected=torch.cat([reference[0,:3],reference[1,:5]])
    torch.testing.assert_close(nested.values(),expected)
    grad,=torch.autograd.grad(nested.values().sum(),logits,retain_graph=True)
    ref_grad,=torch.autograd.grad(expected.sum(),logits)
    torch.testing.assert_close(grad,ref_grad)
    assert grad[batch['padding_mask']].count_nonzero()==0

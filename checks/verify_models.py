"""Known three-node propagation and independent MLP equivalence checks."""
import torch
from src.gcn_layer import GraphConvolution
from src.gcn_model import GCN
from src.mlp_model import MLP
from tools.train import should_stop
def main():
    torch.manual_seed(123)
    a=torch.tensor([[.5,1/(6**.5),0],[1/(6**.5),1/3,1/(6**.5)],[0,1/(6**.5),.5]])
    x=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    layer=GraphConvolution(2,1)
    with torch.no_grad():layer.weight.copy_(torch.tensor([[1.],[2.]]))
    out=layer(x,a.to_sparse())
    torch.testing.assert_close(out.flatten(),torch.tensor([1.3164966,2.2996598,2.3164966]),rtol=1e-5,atol=1e-6)
    out.sum().backward();assert torch.isfinite(layer.weight.grad).all()
    torch.manual_seed(42);g=GCN(2,3,2)
    torch.manual_seed(42);m=MLP(2,3,2)
    g.eval();m.eval()
    assert torch.equal(g.gc1.weight,m.w0) and torch.equal(g.gc2.weight,m.w1)
    torch.testing.assert_close(g(x,torch.eye(3).to_sparse()),m(x),rtol=0,atol=0)
    assert not should_stop(10,[1.]*10+[2.])
    assert should_stop(11,[1.]*11+[2.])
    assert not should_stop(11,[1.]*12)
    print('Known propagation, gradients, initialization, identity equivalence and early-stop boundary passed.')
if __name__=='__main__':main()

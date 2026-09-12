import ast
import unittest
from pathlib import Path

import torch
from esvr4d.model import ESVR4D, rotation4d, query_block


class Mathematics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_rotation_and_upstream(self):
        l,r = torch.randn(7,4,dtype=torch.double),torch.randn(7,4,dtype=torch.double)
        rot = rotation4d(l,r)
        torch.testing.assert_close(rot@rot.transpose(-1,-2),torch.eye(4,dtype=torch.double).expand(7,4,4))
        torch.testing.assert_close(torch.linalg.det(rot),torch.ones(7,dtype=torch.double))
        source = Path('.tools/fudan-4dgs-shallow/utils/general_utils.py')
        if source.exists():
            # Load only the pure rotation function, not upstream's CUDA dependencies.
            tree = ast.parse(source.read_text())
            func = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_rotation_4d')
            namespace = {'torch':torch}
            exec(compile(ast.Module(body=[func],type_ignores=[]),str(source),'exec'),namespace)
            torch.testing.assert_close(rot,namespace['build_rotation_4d'](l,r))

    def test_conditional_factorization_and_gradcheck(self):
        model = ESVR4D(4).double()
        with torch.no_grad():
            model.left.normal_(); model.right.normal_()
        mean,cov,gate = model.condition(0.31)
        x = torch.randn(4,3,dtype=torch.double)
        d4 = torch.cat((x-model.xyz,0.31-model.time),-1)
        joint = torch.exp(-0.5*torch.einsum('ni,nij,nj->n',d4,model.covariance4d().inverse(),d4))
        delta = x-mean
        conditional = torch.exp(-0.5*torch.einsum('ni,nij,nj->n',delta,cov.inverse(),delta))
        torch.testing.assert_close(joint,conditional*gate[:,0])
        self.assertTrue((torch.linalg.eigvalsh(cov)>0).all())
        self.assertTrue(torch.autograd.gradcheck(rotation4d,(model.left,model.right)))

    def test_static_limit_and_temporal_gate(self):
        model=ESVR4D(1,temporal_scale=0.1).double()
        with torch.no_grad(): model.time.fill_(0.5)
        a,c,g=model.condition(0.5)
        b,d,h=model.condition(0.7)
        torch.testing.assert_close(a,b); torch.testing.assert_close(c,d)
        torch.testing.assert_close(h,torch.full_like(h,torch.exp(torch.tensor(-2.)).item()),rtol=1e-6,atol=1e-8)
        self.assertEqual(len(model.at(0).get_xyz),0)

    def test_time_densification_and_optimizer(self):
        model=ESVR4D(4)
        optimizer=torch.optim.Adam(model.parameters())
        sum(p.square().sum() for p in model.parameters()).backward(); optimizer.step()
        old=model.xyz.detach().clone()
        model.observations.fill_(1); model.grad_time[0]=1
        event=model.densify(optimizer,max_points=5,min_scalar=0)
        self.assertEqual(event['split'],1); self.assertEqual(len(model.xyz),5)
        torch.testing.assert_close(model.xyz[:3],old[1:])
        self.assertTrue((optimizer.state[model.xyz]['exp_avg'][3:]==0).all())
        optimizer.zero_grad(); sum(p.square().sum() for p in model.parameters()).backward(); optimizer.step()


@unittest.skipUnless(torch.cuda.is_available(),'CUDA voxelizer requires GPU')
class CUDA(unittest.TestCase):
    def test_covariance_forward_backward_and_tiles(self):
        torch.manual_seed(4)
        m=ESVR4D(2,shape=(12,10,8),temporal_cutoff=0).cuda()
        with torch.no_grad():
            m.xyz.fill_(0.02); m.time.fill_(0.4)
            m.log_scale.copy_(torch.tensor([[-1.2,-1.4,-1.6,-.6]],device='cuda').expand(2,4))
            m.left.add_(torch.randn_like(m.left)*.04)
            m.right.add_(torch.randn_like(m.right)*.04)
        pred=query_block(m,.6,(12,10,8),(0,0,0),(12,10,8))
        tiled=torch.cat([query_block(m,.6,(12,10,8),(i,0,0),(6,10,8)) for i in (0,6)],0)
        torch.testing.assert_close(pred,tiled,atol=2e-6,rtol=2e-5)
        # Independent dense implementation of the existing ESVR CUDA kernel.
        mean,cov,gate=m.condition(.6)
        grid=torch.stack(torch.meshgrid(*[(torch.arange(s,device='cuda')+.5-s/2)*(2/12) for s in (12,10,8)],indexing='ij'),-1)
        delta=grid[None]-mean[:,None,None,None]
        power=torch.einsum('nxyzi,nij,nxyzj->nxyz',delta,cov.inverse(),delta)
        contribution=m.logit_scalar.sigmoid()[:,:,None,None]*gate[:,:,None,None]*torch.sigmoid(5*(1-power))
        reference=torch.where((power<=2)&(contribution>=1/255),contribution,0).sum(0)
        torch.testing.assert_close(pred,reference,atol=2e-6,rtol=2e-5)
        native_grads=torch.autograd.grad(pred.square().sum(),tuple(m.parameters()),retain_graph=True)
        reference_grads=torch.autograd.grad(reference.square().sum(),tuple(m.parameters()))
        for a,b in zip(native_grads,reference_grads):
            torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-4)
        # Restrict finite differences to central voxels away from hard support edges.
        weight=torch.zeros_like(pred); weight[5:7,4:6,3:5]=1
        (pred.square()*weight).sum().backward()
        for name,p in m.named_parameters():
            self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(p.grad.abs().max().item(),0,name)
        # Finite differences away from truncation boundaries; tolerates float atomics.
        for name,index in [('time',(0,0)),('left',(0,1)),('log_scale',(0,3)),('xyz',(0,0))]:
            param=getattr(m,name); analytical=param.grad[index].item(); eps=1e-4
            with torch.no_grad():
                original=param[index].item()
                param[index]=original+eps
                plus=(query_block(m,.6,(12,10,8),(0,0,0),(12,10,8)).square()*weight).sum().item()
                param[index]=original-eps
                minus=(query_block(m,.6,(12,10,8),(0,0,0),(12,10,8)).square()*weight).sum().item()
                param[index]=original
            self.assertAlmostEqual(analytical,(plus-minus)/(2*eps),delta=max(.004,abs(analytical)*.05),msg=name)


if __name__=='__main__': unittest.main()

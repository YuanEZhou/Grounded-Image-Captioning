import torch
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
import pdb
import torch.nn as nn
import torch.nn.functional  as F

class LossWrapper(torch.nn.Module):
	def __init__(self, model, opt):
		super(LossWrapper, self).__init__()
		self.opt = opt
		self.model = model
		if opt.label_smoothing > 0:
			self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
		else:
			self.crit = utils.LanguageModelCriterion()
		self.rl_crit = utils.RewardCriterion()

		if opt.att_supervise:
			if opt.att_sup_crit == 'KL':
				self.kl_crit=nn.KLDivLoss(reduction='batchmean')
			elif opt.att_sup_crit == 'NLL':
				self.nll = nn.NLLLoss()
			elif opt.att_sup_crit == 'ExtendNLL':
				self.extendnll = utils.ExtendNLLCrit()
			else:
				raise NotImplementedError
		self.min_value=1e-8

	def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
				sc_flag,box_inds):
		out = {}
		if not sc_flag:
			if self.opt.att_supervise:
				outputs, attn_weights=self.model(fc_feats, att_feats, labels, att_masks)
				loss1 = self.crit(outputs, labels[:,1:], masks[:,1:])

				if self.opt.use_gt_box:
					box_inds = box_inds[:,1:]
					if self.opt.att_sup_crit == 'KL' or self.opt.att_sup_crit == 'ExtendNLL':
						sup_mask = (box_inds != 1e-8* torch.ones(box_inds.size(-1)).type_as(box_inds)).any(dim=-1).view(-1)
					else:
						sup_mask =  (box_inds>=0).view(-1)
				else:
					_, grd_weights,noun_mask= get_self_critical_reward(self.model, fc_feats, att_feats, att_masks, gts, labels[:,1:].detach(), vars(self.opt))
					sup_mask =  (noun_mask==1).cuda().view(-1)

				attn_weights = torch.log(torch.clamp(attn_weights,min=self.min_value)).view(-1,attn_weights.size(-1))[sup_mask]

				if self.opt.use_gt_box:
					if self.opt.att_sup_crit == 'KL':
						# Todo
						grd_target = F.softmax(box_inds/0.5,dim=-1).view(-1, box_inds.size(-1))[sup_mask]
						loss2 = self.kl_crit(attn_weights, grd_target)
					elif self.opt.att_sup_crit == 'NLL':
						grd_target = box_inds.reshape(-1)[sup_mask].long()
						loss2 = self.nll(attn_weights,grd_target)
					elif self.opt.att_sup_crit == 'ExtendNLL':
						grd_target = box_inds.reshape(-1, box_inds.size(-1))[sup_mask]
						loss2 = self.extendnll(attn_weights, grd_target)
				else:
					if self.opt.att_sup_crit == 'KL':
						grd_target = torch.clamp(grd_weights[:,:17,:],min=self.min_value).view(-1,grd_weights.size(-1))[sup_mask]
						loss2 = self.kl_crit(attn_weights, grd_target)
					elif self.opt.att_sup_crit == 'NLL':
						grd_target = torch.max(grd_weights[:,:17,:],dim=2)[1].view(-1)[sup_mask]
						loss2 = self.nll(attn_weights,grd_target)
					elif self.opt.att_sup_crit == 'ExtendNLL':
						# grd_target = torch.clamp(grd_weights[:,:17,:],min=self.min_value).view(-1,grd_weights.size(-1))[sup_mask]
						# loss2 = self.extendnll(attn_weights, grd_target)
						raise NotImplementedError
				
				loss=loss1+self.opt.att_supervise_weight*loss2
			else:
				outputs=self.model(fc_feats, att_feats, labels, att_masks)[0]
				loss = self.crit(outputs, labels[:,1:], masks[:,1:])
		else:
			if self.opt.att_supervise:
				gen_result, sample_logprobs, attn_weights = self.model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
			else:
				gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
			gts = [gts[_] for _ in gt_indices.tolist()]

			if self.opt.att_supervise:
				reward, grd_weights, noun_mask= get_self_critical_reward(self.model, fc_feats, att_feats, att_masks, gts, gen_result, vars(self.opt))
			else:
				reward = get_self_critical_reward(self.model, fc_feats, att_feats, att_masks, gts, gen_result, vars(self.opt))
			reward = torch.from_numpy(reward).float().to(gen_result.device)

			if self.opt.att_supervise:
				loss1=self.rl_crit(sample_logprobs, gen_result.data, reward)
				sup_mask =  (noun_mask==1).cuda().view(-1)
				attn_weights = torch.log(torch.clamp(attn_weights,min=self.min_value)).view(-1,attn_weights.size(-1))[sup_mask]
				if self.opt.att_sup_crit == 'KL':
					grd_target = torch.clamp(grd_weights,min=self.min_value).view(-1,grd_weights.size(-1))[sup_mask]
					loss2 = self.kl_crit(attn_weights, grd_target)
				elif self.opt.att_sup_crit == 'NLL':
					grd_target = torch.max(grd_weights,dim=2)[1].view(-1)[sup_mask]
					loss2 = self.nll(attn_weights,grd_target)
				elif self.opt.att_sup_crit == 'ExtendNLL':
					# grd_target = torch.clamp(grd_weights,min=self.min_value).view(-1,grd_weights.size(-1))[sup_mask]
					# loss2 = self.extendnll(attn_weights, grd_target)
					raise NotImplementedError

				loss=loss1+self.opt.att_supervise_weight*loss2
			else:
				loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
			out['reward'] = reward[:,0].mean()
		out['loss'] = loss
		return out

import torch 
from config import device, PAD_token, SOS_token, EOS_token, UNK_token

class beam:
	def __init__(self, size, min_length=0, n_best=1):
		self.size = size
		self.n_best = n_best
		self.min_length = min_length

		# backpointers at each step
		self.prev_ps = []
		# next predicted token at each step
		self.next_ts = [torch.LongTensor(size).fill_(PAD_token).to(device)]
		self.next_ts[0][0] = SOS_token
		# score for each path on the beam
		self.scores = torch.zeros(size, dtype=torch.float, device = device)
		self.all_scores = [] # all histories
		self.n_best = n_best # choose how many paths finally

		self.finish_paths = [] # all paths from the beam
		self.stopByEOS = False

	def advance(self, words_probs):
		'''
		words_probs: k*vocab_size
		'''
		vocab_size = words_probs.size(1)
		cur_len = len(self.prev_ps)
		if cur_len < self.min_length:
			for i in range(words_probs.size(0).item()):
				words_probs[i, EOS_token] = -1e20

		if cur_len == 0:
			step_scores = words_probs[0]
		else:
			step_scores = words_probs + self.scores.unsqueeze(1).expand_as(words_probs)
		step_scores_flatten = step_scores.view(-1)
		best_scores, best_scores_tokenId = step_scores_flatten.topk(self.size)
		self.scores = best_scores
		
		prev_step = best_scores_tokenId/vocab_size
		self.prev_ps.append(prev_step)
		self.next_ts.append(best_scores_tokenId - prev_step*vocab_size)

		next_token = self.next_ts[-1]
		if next_token[0] == EOS_token:
			self.stopByEOS = True

		for k in range(self.size):
			if next_token[k] == EOS_token:
				self.finish_paths.append((best_scores[k], len(self.prev_ps), k))

		# if len(self.prev_ps) == tgt_max_length:
		# 	self.stopByMAX = True 

		return None

	def get_pred_sentence(self, sent_info):
		score, step_stop, path_idx = sent_info
		sent_token = []
		for i in range(step_stop-1, -1, -1):
			sent_token.append(self.next_ts[i+1][path_idx].item())
			path_idx = self.prev_ps[i][path_idx]
		return score, sent_token[::-1]
		









    

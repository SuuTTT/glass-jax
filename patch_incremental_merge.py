import re

with open("src/glass/seclust/incremental.py", "r") as f:
    content = f.read()

new_merge_code = """

    def merge_delta(self, left: int, right: int, weight_between: float) -> float:
        \"\"\"Calculate flat SE delta for merging two clusters.\"\"\"
        if left == right:
            return 0.0
            
        vol_L = self.volume[left]
        cut_L = self.cut[left]
        dlogd_L = self.degree_log_degree[left]
        
        vol_R = self.volume[right]
        cut_R = self.cut[right]
        dlogd_R = self.degree_log_degree[right]
        
        new_vol = vol_L + vol_R
        new_cut = cut_L + cut_R - 2.0 * weight_between
        new_dlogd = dlogd_L + dlogd_R
        
        old_entropy_L = self._compute_cluster_entropy(vol_L, cut_L, dlogd_L)
        old_entropy_R = self._compute_cluster_entropy(vol_R, cut_R, dlogd_R)
        new_entropy = self._compute_cluster_entropy(new_vol, new_cut, new_dlogd)
        
        return new_entropy - old_entropy_L - old_entropy_R

    def apply_merge(self, left: int, right: int, weight_between: float) -> float:
        \"\"\"Apply a merge and return the entropy change.\"\"\"
        delta = self.merge_delta(left, right, weight_between)
        
        self.volume[left] += self.volume[right]
        self.cut[left] += self.cut[right] - 2.0 * weight_between
        self.degree_log_degree[left] += self.degree_log_degree[right]
        self.size[left] += self.size[right]
        
        self.volume[right] = 0.0
        self.cut[right] = 0.0
        self.degree_log_degree[right] = 0.0
        self.size[right] = 0
        self.active[right] = False
        
        self.labels[self.labels == right] = left
        self.entropy += delta
        if abs(self.entropy) < 1e-12:
            self.entropy = 0.0
        return delta
"""

old_apply_move = """        self.entropy += delta
        if abs(self.entropy) < 1e-12:
            self.entropy = 0.0
        return delta"""

content = content.replace(old_apply_move, old_apply_move + new_merge_code)

with open("src/glass/seclust/incremental.py", "w") as f:
    f.write(content)

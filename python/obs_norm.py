import numpy as np
 
 
class RunningNorm:
 
    def __init__(self, size, clip=10.0, epsilon=1e-4):
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = epsilon
        self.clip = clip
 
    def update(self, batch):
        """batch: array of shape (n, size). Call once per observation."""
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[None, :]
 
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
 
        delta = batch_mean - self.mean
        total = self.count + batch_count
 
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
 
        self.mean = new_mean
        self.var = m2 / total
        self.count = total
 
    def normalize(self, x):
        """(x - mean) / std, then clipped so one crazy outlier cannot
        produce a huge activation."""
        x = np.asarray(x, dtype=np.float64)
        z = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)
 
    # ---- checkpointing ----
 
    def state_dict(self):
        return {"mean": self.mean, "var": self.var,
                "count": self.count, "clip": self.clip}
 
    def load_state_dict(self, d):
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.var = np.asarray(d["var"], dtype=np.float64)
        self.count = float(d["count"])
        self.clip = float(d.get("clip", 10.0))

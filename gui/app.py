import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import numpy as np
from torchvision import transforms

from config import EMBED_DIM, INDEX_TYPE, INDEX_PARAMS, DATA_DIR
from model.embedding_model import EmbeddingNet
from indexer.faiss_index import FaissIndex
from indexer.hnsw_index import HNSWIndex

class PlushieGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Plushie Recognizer")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.index = None
        self.image_path = None

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(padx=10, pady=10, fill="x")

        tk.Button(btn_frame, text="加载模型", command=self.load_model).pack(side="left", padx=5)
        tk.Button(btn_frame, text="加载索引", command=self.load_index).pack(side="left", padx=5)
        tk.Button(btn_frame, text="选择图片", command=self.select_image).pack(side="left", padx=5)

        tk.Label(btn_frame, text="Top K:").pack(side="left", padx=(20,0))
        self.k_var = tk.IntVar(value=5)
        tk.Entry(btn_frame, width=4, textvariable=self.k_var).pack(side="left")

        tk.Button(btn_frame, text="预测", command=self.predict).pack(side="left", padx=5)

        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=10, pady=5)

        self.result_text = tk.Text(self.root, height=10, width=50)
        self.result_text.pack(padx=10, pady=5)

    def load_model(self):
        path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch 模型", "*.pt"), ("所有文件", "*.*")]
        )
        if not path:
            return
        self.model = EmbeddingNet().to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        messagebox.showinfo("提示", f"模型已加载：\n{path}")

    def load_index(self):
        dirpath = filedialog.askdirectory(
            title="选择索引目录（包含 .idx/.labels 文件）"
        )
        if not dirpath:
            return
        if INDEX_TYPE == 'faiss':
            idx = FaissIndex(dim=EMBED_DIM,
                             nlist=INDEX_PARAMS['nlist'],
                             nprobe=INDEX_PARAMS['nprobe'])
        else:
            idx = HNSWIndex(dim=EMBED_DIM)
        idx.load(os.path.join(dirpath, "main_index"))
        self.index = idx
        messagebox.showinfo("提示", f"索引已加载：\n{dirpath}")

    def select_image(self):
        path = filedialog.askopenfilename(
            title="选择查询图片",
            filetypes=[("图片", "*.jpg;*.jpeg;*.png"), ("所有文件", "*.*")]
        )
        if not path:
            return
        self.image_path = path
        img = Image.open(path)
        img.thumbnail((300,300))
        photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # 保持引用

    def predict(self):
        if self.model is None:
            messagebox.showwarning("警告", "请先加载模型")
            return
        if self.index is None:
            messagebox.showwarning("警告", "请先加载索引")
            return
        if not self.image_path:
            messagebox.showwarning("警告", "请先选择图片")
            return

        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        img = Image.open(self.image_path).convert('RGB')
        tensor = tf(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(tensor).cpu().numpy()[0]
        results = self.index.search(emb, self.k_var.get())

        self.result_text.delete("1.0", tk.END)
        for rank, (label, dist) in enumerate(results, start=1):
            self.result_text.insert(tk.END, f"{rank}. {label} (距离 {dist:.4f})\n")

    def run(self):
        self.root.mainloop()

import torch
import torch.nn as nn
from transformers import AutoModel
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from training import model_setup
import process
from cross_attn_fusion import ImprovedAttentionCombiner

class CombinedEmbeddingModel(nn.Module):
        def __init__(self, transformer_name, supervised_model, supervised_dim, graph_level=False, combined_dim=512):
            super(CombinedEmbeddingModel, self).__init__()
            self.device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Loading transformer.. {transformer_name}")
            self.transformer = AutoModel.from_pretrained(transformer_name)
            self.transformer = self.transformer.to(self.device)
            self.supervised_model = supervised_model.to(self.device)  # Pretrained supervised model
            self.graph_level = graph_level
            self.supervised_model.get_node_embedding = not graph_level
            self.supervised_model.get_graph_embedding = graph_level
            self.supervised_proj = nn.Linear(supervised_dim, combined_dim).to(self.device)
            self.transformer_proj = nn.Linear(self.transformer.config.hidden_size, combined_dim).to(self.device)
            self.attention_combiner = ImprovedAttentionCombiner(combined_dim).to(self.device)
            self.fc = nn.Linear(combined_dim, 1).to(self.device)  # Output layer for regression
            

        def forward(self, text, input_for_supervised, tokenizer):
            # Get transformer embeddings
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs=inputs.to(self.device)
            transformer_output = self.transformer(**inputs).last_hidden_state.mean(dim=1)
            # outputs = self.transformer(**inputs)
            # transformer_output = outputs.pooler_output 
            transformer_proj = self.transformer_proj(transformer_output)

            # Get supervised model embeddings
            input_for_supervised = input_for_supervised.to(self.device)
            _, supervised_embedding = self.supervised_model(input_for_supervised)
            if self.graph_level:
                supervised_proj = self.supervised_proj(supervised_embedding)
                # Combine embeddings with attention
                combined_embedding = self.attention_combiner(supervised_proj.unsqueeze(1), transformer_proj.unsqueeze(1))
            else: 
                supervised_embedding, node_mask = to_dense_batch(supervised_embedding,input_for_supervised.batch)
                supervised_proj = self.supervised_proj(supervised_embedding)
                # Combine embeddings with attention
                combined_embedding = self.attention_combiner(supervised_proj, transformer_proj.unsqueeze(1), node_mask)

            # Predict property
            output = self.fc(combined_embedding)
            return output

class SupervisedModel():
        def __init__(self, training_parameters, model_parameters, job_parameters, processing_parameters):
            self.training_parameters = training_parameters
            self.model_parameters = model_parameters
            self.job_parameters = job_parameters
            self.processing_parameters =processing_parameters

        def get_data(self, data_path):
            dataset = process.get_dataset(data_path=data_path, target_index=self.training_parameters["target_index"],
                                          reprocess=True, model_name=self.model_parameters["model"], processing_args= self.processing_parameters)
            print(f"Type of dataset_cif: {type(dataset)}")
            print(f"First item type: {type(dataset[0]) if hasattr(dataset, '__getitem__') else 'Not indexable'}")
            print(f"Dataset length: {len(dataset)}")
            return dataset

        def load_data(self, dataset):
            train_sampler = None
            loader = DataLoader(
                dataset,
                batch_size=self.model_parameters["batch_size"],
                shuffle=(train_sampler is None),
                num_workers=0,
                pin_memory=False,
                sampler=train_sampler,
            )
            return loader

        def load_model(self, dataset, rank='cuda'):
            print(f"Loading model.. {self.model_parameters['model']}")
            ##Set up model
            model = model_setup(
                rank,
                self.model_parameters["model"],
                self.model_parameters,
                dataset,
                False,
                # self.job_parameters["load_model"],
                # self.job_parameters["model_path"],
                "./Results/Models/cgcnn.pth",
                self.model_parameters.get("print_model", False),
            )
    
            return model

from .wrapper_layer import WrapperLayer
from .wrapper_layer import BloomWrapperLayer, LlamaWrapperLayer
from .sparsity_util import find_layers, sparsify_matrix_for_FC_layer
import torch
import os

#PruneMetadata is used to store the statistics during the forward pass of the model.
class PruneMetadata:
    def __init__(self, model, output_path=None, enable_weight_wise_pruning=True):
        self.all_wrapper_layers = []
        self.handles = []
        self.model = model
        self.output_path = output_path
        self.enable_weight_wise_pruning = enable_weight_wise_pruning
        self.sparsity_percentage = 1.0

    def register_hooks_for_layers(self, layers):
        for id, layer in enumerate(layers):
            subset = self.find_instrument_layers(layer)
        
            # Wrapper layer is used to record the statistics of each layer
            wrapper_layers = {}
            for name in subset:
                wrapper_layers[name] = self.create_wrapper_layer(subset[name], layer_id=id, layer_name=name)
            self.all_wrapper_layers.append(wrapper_layers)
            
            def add_batch(layer_id, name, wrapper_layer):
                def tmp(_, inp, out):
                    # print('[DEBUG-0]layer_id:{}, layer_name:{}'.format(layer_id, name))
                    wrapper_layer.add_batch(inp[0].data, out.data)
                    # print('[DEBUG-1]layer_id:{}, layer_name:{}'.format(layer_id, name))
                return tmp

            def prune_weights(layer_id, name, wrapper_layer: WrapperLayer):
                def prune(module, input, output:torch.tensor):
                    print(layer_id, name, wrapper_layer.layer_name, self.output_path)
                    _, seq_len, d_model = output.size()
                    filename = f"{layer_id}_{name}.pt"
                    # Load the activations from previously recorded files
                    res = torch.load(os.path.join(self.output_path, filename))
                    assert res.size() == output.size()
                    pruned_res = sparsify_matrix_for_FC_layer(res, self.sparsity_percentage, seq_len * d_model) > 0
                    return output * pruned_res
                    
                return prune
            for name, wrapper_layer in wrapper_layers.items():
                if self.enable_weight_wise_pruning:
                    # prune weight based on recorded activation information
                    self.handles.append(subset[name].register_forward_hook(prune_weights(id, name, wrapper_layer)))
                else:
                    # record activation information
                    self.handles.append(subset[name].register_forward_hook(add_batch(id, name, wrapper_layer)))                

    def find_instrument_layers(self, layer):
        return find_layers(layer)
    
    def create_wrapper_layer(self, layer, layer_id, layer_name):
        return WrapperLayer(layer, layer_id, layer_name)

    def print(self, save_weight_importance=True):
        print("PruneMetadata")
        print("all_wrapper_layers:")
        for id, wrapper_layers in enumerate(self.all_wrapper_layers):
            print(" layer_id:", id)
            for name, wrapper_layer in wrapper_layers.items():
                print("  layer_name:", name)
                print("    rows:", wrapper_layer.rows)
                print("    columns:", wrapper_layer.columns)
                print("    nsamples:", wrapper_layer.nsamples)
                print("    scaler_row.shape:", wrapper_layer.scaler_row.shape)
                weight_importance = wrapper_layer.get_weight_importance()
                print("    weight_importance.shape:", weight_importance.shape)
                if self.output_path is not None:
                    if not os.path.exists(self.output_path):
                        os.makedirs(self.output_path)
                    filename = f"{id}_{name}.pt"
                    torch.save(weight_importance, os.path.join(self.output_path, filename))
        
# TODO: implement this
class BloomPruneMetadata(PruneMetadata):
    def __init__(self, model, output_path):
        super().__init__(model, output_path)
        self.output_path = output_path
        
    def create_wrapper_layer(self, layer, layer_id, layer_name):
        return BloomWrapperLayer(layer, layer_id, layer_name)
    
class LlamaPruneMetadata(PruneMetadata):
    def __init__(self, model, activation_func, output_path):
        super().__init__(model, output_path)
        self.activation_func = activation_func
        
    def create_wrapper_layer(self, layer, layer_id, layer_name):
        return LlamaWrapperLayer(layer, layer_id, layer_name, self.activation_func)
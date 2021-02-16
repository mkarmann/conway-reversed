import torch
import torchvision
import os

from bestguess.bestguess import create_random_board, create_net_input_array, BestGuessModule, MODEL_CHANNELS


class ExportBestGuessModel(BestGuessModule):

    def forward(self, x):
        return self.get_probabilities(x)


if __name__ == '__main__':
    device = torch.device('cuda')
    dummy_state = create_random_board(shape=(6, 6), warmup_steps=5)

    dummy_input_tensor = torch.from_numpy(create_net_input_array(dummy_state, dummy_state, dummy_state)).unsqueeze(0).float().to(device)
    net = ExportBestGuessModel(channels=MODEL_CHANNELS)
    net.load_state_dict(torch.load(
        'P:\\python\\convay-reversed\\best_guess\\out\\training\\snapshots\\128___2020_11_16_04_43_05_GoL_delta_1continue\\epoch_040.pt'))
    net.to(device).eval()

    torch.onnx.export(net, dummy_input_tensor, "../out/model.onnx", verbose=True, output_names=['output.1'])
    # os.system('onnx-tf convert -i ./out/model.onnx -o ./out/exported.pb')

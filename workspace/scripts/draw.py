import matplotlib.pyplot as plt
import os
import json
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('f', '', 'log file location')


class Data(object):
    def __init__(self) -> None:
        self.iters = []
        self.action = []
        self.gripper = []
        self.predict = []
        self.adapt_loss = []

    def parse(self, line: str):
        iter, line = line.split(maxsplit=1)
        if not iter.isdigit() or 'model-saved' in line or 'new-epoch' in line or int(iter) == 0:
            return
        iter = int(iter)
        if len(self.iters) == 0 or self.iters[-1] != iter:
            self.iters.append(iter)
        msg, line = line.split(maxsplit=1)
        if 'adapt-loss' in msg:
            assert self.iters[-1] == iter
            self.adapt_loss.append(json.loads(line))
        elif 'meta-loss' in msg:
            assert self.iters[-1] == iter
            loss = json.loads(line)
            self.action.append(loss[0])
            self.gripper.append(loss[1])
            self.predict.append(loss[2])

    def __getitem__(self, idx):
        return self.iters[idx], self.action[idx], self.gripper[idx], self.predict[idx], self.adapt_loss[idx]

    def __len__(self):
        # print(len(self.iters), len(self.action), len(self.gripper), len(self.predict), len(self.adapt_loss))
        assert len(self.iters) == len(self.action) == len(
            self.gripper) == len(self.predict) == len(self.adapt_loss)
        return len(self.iters)

    def draw(self):
        plt.plot(self.iters, self.action, label='continuous')
        plt.plot(self.iters, self.gripper, label='discrete')
        plt.plot(self.iters, self.predict, label='predict')
        plt.plot(self.iters, [a+g+p for a, g, p in zip(self.action,
                 self.gripper, self.predict)], label='meta-loss')
        plt.plot(self.iters, [sum(adapt) / len(adapt)
                 for adapt in self.adapt_loss], label='adapt-loss')
        plt.fill_between(self.iters, [min(adapt) for adapt in self.adapt_loss], [
                         max(adapt) for adapt in self.adapt_loss], alpha=0.2, color='purple')
        plt.xlabel('iteration')
        plt.xlim([0, self.iters[-1]])
        plt.ylabel('loss')
        plt.ylim([0, 15])
        plt.legend(loc='best')
        plt.show()


def main(argv):
    file_path = FLAGS.f
    assert os.path.exists(file_path)
    data = Data()
    file = open(file_path, 'r')
    print(file.readline())
    for line in file.readlines():
        line = line.split(']| ', 1)[-1]
        data.parse(line)

    file.close()

    print(len(data))
    data.draw()
    # for i in range(len(data)):
    #     print(data[i])


if __name__ == '__main__':
    app.run(main)

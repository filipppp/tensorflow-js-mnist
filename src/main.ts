import * as tf from '@tensorflow/tfjs-node';
import {NeuronalNetwork} from "./NeuronalNetwork";

setInterval(() => (console.log(tf.memory())), 3000)

let nn = new NeuronalNetwork(tf.losses.meanSquaredError, 0.1, "sigmoid", "model/","model/model.json");

test();

async function test() {
  await nn.loadModel();
  await nn.train();
  console.log(nn.test());
}

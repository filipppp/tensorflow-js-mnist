import * as tf from '@tensorflow/tfjs-node';
import {NeuronalNetwork} from "./NeuronalNetwork";


let nn = new NeuronalNetwork(tf.losses.meanSquaredError, 0.1, "sigmoid", "model/","model/model.json");

test();

async function test() {
  await nn.loadModel();
  console.log(nn.test());
}

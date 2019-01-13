import * as tf from "@tensorflow/tfjs-node";
import {Model, Tensor} from '@tensorflow/tfjs-node';
import * as mnist from 'mnist';

export class NeuronalNetwork{

  public trainingInput: Tensor;
  public trainingOutput: Tensor;
  public testInput: Tensor;
  public testOutput: Tensor;

  private _model: Model;
  private _data;
  private _compileArgs;

  constructor(private _lossFunction: Function,
              private _learningRate: number,
              private _activation: string,
              public savePath: string,
              private modelPath?: string) {
    this._compileArgs = {
      optimizer: tf.train.sgd(_learningRate),
      loss: _lossFunction
    };

    if (!modelPath) {
      this._model = this.createModel();
      this._model.compile(this._compileArgs);
    }
    this._data = mnist.set(3000, 300);
    this.generateTestMNIST();
    this.generateTrainingMNIST();
  }

  public test(): string {
    const output: Tensor = this._model.predict(this.testInput);

    const finalOutput = output.dataSync();
    const expectedOutput = this.testOutput.dataSync();

    let string = "";

    let finalTemp =  [];
    let expectedTemp =  [];
    for (let i = 0; i < finalOutput.length; i++) {
      finalTemp.push(finalOutput[i]);
      expectedTemp.push(expectedOutput[i]);
      if(i % 10 === 0 && i !== 0) {
        string += (`
        Excpected Output: ${expectedTemp}
        Final Output:     ${finalTemp}
        `);
        finalTemp = [];
        expectedTemp = [];
      }
    }
    return string;
  }

  public async loadModel() {
    try {
      this._model = await tf.loadModel(`file://${this.modelPath}`);
      this._model.compile(this._compileArgs);
    } catch (e) {
      console.error(e);
    }
  }

  public generateTestMNIST() {
    const testInputs = [];
    const testOutputs = [];
    this._data.test.forEach(oneTest => {
      testInputs.push(oneTest.input);
      testOutputs.push(oneTest.output)
    });
    this.testInput = tf.tensor2d(testInputs);
    this.testOutput = tf.tensor2d(testOutputs);
  }

  public generateTrainingMNIST() {
    const trainingInputs = [];
    const trainingOutputs = [];
    this._data.training.forEach(oneTraining => {
      trainingInputs.push(oneTraining.input);
      trainingOutputs.push(oneTraining.output)
    });
    this.trainingInput = tf.tensor2d(trainingInputs);
    this.trainingOutput = tf.tensor2d(trainingOutputs);
  }

  public async train() {
    if(!this._model) {return;}
    const config = {
      shuffle: true,
      epochs: 1000,
      callbacks: {
        onEpochEnd: async (_, l) => (console.log(l.loss))
      }
    };
    await this._model.fit(this.trainingInput, this.trainingOutput, config);
    await this._model.save(`file://${this.savePath}`)
  }

  private createModel(): Model  {
    return tf.tidy(() => {
      const input = tf.input({
        shape: [784]
      });
      const denseLayer1 = tf.layers.dense({
        units: 30,
        activation: this._activation
      });
      const denseLayer2 = tf.layers.dense({
        units: 30,
        activation: this._activation
      });
      const output = tf.layers.dense({
        units: 10,
        activation: this._activation
      });
      return tf.model({
        inputs: input,
        outputs: output.apply(denseLayer2.apply(denseLayer1.apply(input)))
      });
    });
  }

}

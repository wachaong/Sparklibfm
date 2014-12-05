/**
 */
//package z.fm

import java.io.PrintWriter
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

import breeze.linalg.{norm => brzNorm}
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._

import scala.collection.immutable.SortedMap

case class ClkImpInstance(var click: Int, var impression: Int, var features: BSV[Double])

class FMWithLBFGS(val l2RegParam: Double,
				  val memParam: Int,
				  val factor: Int,
				  val maxNumIterations: Int,
				  val tolerance: Double) extends Serializable {

  def train(trainSet: RDD[ClkImpInstance], initialWeights: BDV[Double]) = {

//	trainSet.cache()

	val costFun = new CostFun(trainSet, l2RegParam)

	val lbfgs = new LBFGS[BDV[Double]](maxNumIterations, memParam, tolerance)

	val states = lbfgs
	  .iterations(new CachedDiffFunction(costFun), initialWeights)


	var state = states.next()
	while (states.hasNext) {
	  state = states.next()
	  println("iter:"+state.iter)
	}
	val weights = state.x
	weights
  }

  private class CostFun(data: RDD[ClkImpInstance],
						l2RegParam: Double) extends DiffFunction[BDV[Double]] with Serializable{

	def calGradientLossInstance(labeledInstance: ClkImpInstance,
								weightsBC: Broadcast[BDV[Double]]): (BSV[Double], Double) = {
	  def score2LossProb(score: Double) = {
		if (score < -30) {
		  (-score, 0.0)
		} else if (score > 30) {
		  (0.0, 1.0)
		} else {
		  val tmp = 1 + math.exp(-score)
		  (math.log(tmp), 1.0 / tmp)
		}
	  }

	  def wTx(weights:BDV[Double], xarray: Array[(Int,Double)], xlength: Int) = {
		val b = weights(weights.length-1)
		val xw = xarray.map(p => weights(p._1)*p._2).sum
		var xv = 0.0
		for(f <- 0 until factor) {
		  val g = xarray.map(p => {
			val (i,xi) = p
			val vif = weights(xlength+factor*i+f)
			val vx = vif*xi
			(vx,vx*vx)
		  }).reduce((x,y) =>(x._1+y._1,x._2+y._2))
		  xv += g._1*g._1 - g._2
		}
		b + xw + 0.5*xv
	  }

	  val weights = weightsBC.value
	  val (clicks, imps) = (labeledInstance.click, labeledInstance.impression)
	  val nonclicks = imps - clicks

	  val x = labeledInstance.features
	  val xarray = x.activeIterator.toArray
	  val xlength = x.length

	  val score = wTx(weights,xarray,xlength)

	  var totalMult = 0.0
	  var totalLoss = 0.0

	  if (clicks > 0) {
		val (loss, prob) = score2LossProb(score)
 		val mult = (prob - 1.0) * clicks
		totalMult = mult
		totalLoss = loss * clicks
	  }

	  if (nonclicks > 0) {
		val (loss, prob) = score2LossProb(-1*score)
		val mult = (1.0 - prob) * nonclicks
		totalMult += mult
		totalLoss += loss * nonclicks
	  }



	  val indices = new ArrayBuffer[Int]()
	  val values = new ArrayBuffer[Double]()
	  indices ++= x.index
	  values ++= x.data

	  for(f <- 0 until factor) {
		val sumvxf = xarray.map(p => {
		  val (i, xi) = p
		  val vif = weights(xlength+factor*i+f)
		  vif*xi
		}).sum
		xarray.map(p => {
		  val (i, xi) = p
		  val vif = weights(xlength+factor*i+f)
		  indices += xlength+factor*i+f
		  values += xi*sumvxf - vif*xi*xi
		})
	  }

	  indices += weights.length - 1
	  values += 1

	  val (sortedIndices,sortedValues) = (indices zip values).sortBy(_._1).unzip

	  val gradient = new BSV(sortedIndices.toArray,sortedValues.toArray, weights.length) * totalMult
	  (gradient, totalLoss)

	}

	override def calculate(x: BDV[Double]): (Double, BDV[Double]) = {
	  val wb = data.sparkContext.broadcast(x)

	  val kvs = data.flatMap(inst => {
		val (grad, loss) = calGradientLossInstance(inst, wb)
		grad.activeIterator.toSeq :+(-1, loss)
	  }).reduceByKeyLocally(_ + _)

	  val gradient = new Array[Double](x.length)
	  for (index <- kvs.filterKeys(_ >= 0).keys)
		gradient(index) = kvs.getOrElse(index, 0.0) + l2RegParam * x(index)

//	  gradient.foreach(println(_))

	  val norm = brzNorm(x,2)
	  val loss = kvs(-1) + 0.5 * l2RegParam * norm * norm

	  (loss, new BDV(gradient))

	}
  }

}

object FactorizationMachine {

  def line2Vector(content:String, length: Int) = {
	val indices = new ArrayBuffer[Int]()
	val values = new ArrayBuffer[Double]()

	for(field <- content.split("\\p{Blank}+")) {
	  val kv = field.split(":")
	  if(kv.length == 2) {
		indices += kv(0).toInt
		values += kv(1).toDouble
	  }
	}

	val (sortedIndices,sortedValues) = (indices zip values).sortBy(_._1).unzip
	val bsv = new BSV[Double](sortedIndices.toArray,sortedValues.toArray,length)

	bsv
  }

  def line2ClkImpInstance(content:String,length:Int): ClkImpInstance = {

	val items = content.split("\\p{Blank}+",3)
	val clk = items(0).toInt
	val imp = items(1).toInt
	val features = items(2)

	ClkImpInstance(clk, imp, line2Vector(features,length))
  }

  def main(args: Array[String]): Unit = {
	val conf = new SparkConf().setAppName("FMTrain")
	val sc = new SparkContext(conf)

//	Logger.getRootLogger.setLevel(Level.WARN)

	val input = args(0)
	val length = args(1).toInt
	val factor = args(2).toInt
	val iterNum = args(3).toInt
	val modelOutput = args(4)

	val initWeights = BDV.zeros[Double](length+length*factor+1)

	val random = new Random()
	for(i <- length until initWeights.length - 1) {
	  initWeights(i) = random.nextGaussian() * 0.01
	}

	val rawText = sc.textFile(input)
	val trainset = rawText.map(x=> line2ClkImpInstance(x,length)).cache()

	val model = new FMWithLBFGS(1,10,factor,iterNum,1E-9).train(trainset,initWeights)

	//val out = new PrintWriter(modelOutput)
	//for (index <- model.keysIterator) {
	//  out.println(index + "\t" + model(index))
	//}

        var map = SortedMap[Int,String]()
        for (index <- model.keysIterator) {
                map += index -> (index+"\t"+model(index).toString)
        }
        val arr = map.values.toArray
        val t = sc.parallelize(arr)
        t.saveAsTextFile(modelOutput)
	sc.stop()
  }
}

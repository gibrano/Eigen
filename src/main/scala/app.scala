package descomposition

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{Vector,Vectors}
import org.apache.log4j.{Level, Logger}

object Eigen {
  def main(args: Array[String]) {
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    val conf = new SparkConf(true).setAppName("Eigen Descomposition")
    val sc = new SparkContext(conf)
    var L = Array[org.apache.spark.mllib.linalg.Vector]()
    for( i <- 1 to 100){
      var v = Array[Double]()
      for( j <- 1 to 100){
        if(i == j){
           v = v ++ Array(3.00)
        } else{
           v = v ++ Array(-1.00)
        }
      }
      L = L ++ Array(Vectors.dense(v))
    }

    val t1 = System.nanoTime
    val eigen = Decomposition.eigenValues(L,sc)
    val duration = (System.nanoTime - t1) / 1e9d
    print("Duration Time:",duration, "Numbers of Cores", sc.getExecutorStorageStatus.length)
    sc.stop()
  }
}


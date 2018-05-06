package descomposition

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{Vector,Vectors}

object Decomposition {
  
  def rotate(A: Array[org.apache.spark.mllib.linalg.Vector], k1: Int, k2: Int): Array[org.apache.spark.mllib.linalg.Vector] = {
    val n = A.size - 1
    var w = (A(k2)(k2) - A(k1)(k1))/(2*A(k1)(k2))
    var t = 0.00
    if(w>=0){
      t = -w+math.sqrt(w*w+1)
    } else {
      t = -w-math.sqrt(w*w+1)
    }
    val c = 1/(math.sqrt(1+t*t))
    val s = t/(math.sqrt(1+t*t))
    for(j <- 0 to n){
      var row1 = c*A(k1)(j) - s*A(k2)(j)
      var row2 = s*A(k1)(j) + c*A(k2)(j)
      A(k1).toArray(j) = row1
      A(k2).toArray(j) = row2
    }  
    for(j <- 0 to n){
      var col1 = c*A(j)(k1) - s*A(j)(k2)
      var col2 = s*A(j)(k1) + c*A(j)(k2)
      A(j).toArray(k1) = col1
      A(j).toArray(k2) = col2
    }  
    return A
  }
  
  def pivot(A: Array[org.apache.spark.mllib.linalg.Vector]): Array[Int] = {
    var i = 0
    var j = 1
    val n = A.size - 1
    for( k1 <- 0 to (n-1)){
      for(k2 <- (k1+1) to n){
        if(math.abs(A(i)(j)) < math.abs(A(k1)(k2))){
           i = k1
           j = k2
        }
      }
    }
    return Array(i,j)
  }

  def getError(x1: Array[Double], x2: Array[Double], sc: SparkContext): Double = {
    val n = x1.size - 1
    var index = sc.parallelize(0 to n)
    var diff = index.map(i => math.abs(x2(i) - x1(i)))
    var err = diff.max
    return err
  }

  def getEigen(D: Array[org.apache.spark.mllib.linalg.Vector],  sc: SparkContext): Array[Double] = {
    val n = D.size - 1
    var index = sc.parallelize(0 to n)
    var eigenvalues = index.map(i => D(i)(i)).collect
    return eigenvalues
  }

  def eigenValues(A: Array[org.apache.spark.mllib.linalg.Vector], sc: SparkContext): Array[Double] = {
    var D = A
    var eigen2 = getEigen(D,sc)
    var err = 1.00
    while(err > 0.001){
      var eigen1 = getEigen(D,sc)
      var x = pivot(D)
      D = rotate(D, x(0), x(1))
      eigen2 = getEigen(D,sc)
      err = getError(eigen1,eigen2,sc)
    }
    return eigen2
  }

}


import cern.colt.matrix.DoubleMatrix2D
import cern.colt.matrix.impl.DenseDoubleMatrix2D
import cern.colt.matrix.linalg.Algebra
import cern.jet.math.Functions

import scala.io.Source

object TryColt extends App {

  def computeCostMulti(X: Matrix2D, y: Matrix2D, theta: Matrix2D): Double = {
    //    %   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    //    %   parameter for linear regression to fit the data points in X and y
    //
    assert(theta.rows == X.cols)
    assert(theta.cols == 1)
    assert(y.rows == X.rows)
    assert(y.cols == 1)

    // Initialize some useful values
    // number of training examples
    val m = y.rows
    // ====================== YOUR CODE HERE ======================
    // Instructions: Compute the cost of a particular choice of theta
    //                 You should set J to the cost.


    println("shape(X), shape(theta) = " + X.shape + " " + theta.T.shape)
    val dif0 = X * theta
    println("dif0: " + dif0.shape)
    val dif = dif0 - y
    val Jx0: Matrix2D = dif.T * dif
    val Jx = Jx0 * (0.5 / m.toDouble)
    // =========================================================================
    // You need to return variable J correctly
    Jx(0,0)
  }



  def loadMatrix(path: String) = {
    val text = Source.fromURL(getClass.getResource(path)).getLines()
    val data: Seq[Seq[Double]] = text.map(
      line => line.split(",").map(_.toDouble).toSeq
    ).toSeq
    Matrix2D.fromData(data)
  }


  def part3(ex1data1: Matrix2D): Unit = {
    val columnOfOnes = Matrix2D.ones(ex1data1.rows, 1)
    val tmp: Matrix2D = ex1data1(::, 0 to 1)
    val X = Matrix2D.hstack(columnOfOnes, tmp)
    println("X.shape: " + X.shape)
    val y = ex1data1(::, 1 to 2)
    println("y.shape: " + y.shape)

    val m = X.cols
    val n = X.rows

    // initialize fitting parameters
    val theta = Matrix2D.zeros(m, 1)
    // Some gradient descent settings
    val num_iters = 1500
    val alpha = 0.01

    println("computeCost = " + computeCostMulti(X, y, theta))

    val theta_res = gradientDescent(X, y, theta, alpha, num_iters)
    print("theta_res =", theta_res(::, 0 until 1))
    val cost_res = computeCostMulti(X, y, theta_res)
    print("cost_res =", cost_res)
  }

  def gradientDescent(X: Matrix2D, ymat: Matrix2D, theta0: Matrix2D, alpha: Double, iterations: Int): Matrix2D = {
    //   create a copy of theta for simultaneous update.
    val theta: Matrix2D = theta0
    val m = X.cols
    //   number of features.
    val p = X.rows

    //   ====================== YOUR CODE HERE ======================
    // Instructions: Perform a single gradient step on the parameter vector theta.
    //  Hint: While debugging, it can be useful to print out the values# % of the cost function(computeCost) and gradient here.
    //  simultaneous update theta using theta_prev.
    for (i <- 0 until iterations) {
      //     calculate dJ / d(theta_j)
      for (j <- 0 until p) {
        // TODO
        val deriv: Matrix2D = (X * theta - ymat).T * X(::, j to j) * (1.0 / m)
        val tmp: Matrix2D = theta(j) - deriv * alpha
        theta.update(j, tmp)

        //        % ============================================================
      }
    }
    theta
  }

def main(): Unit = {
    val mat = loadMatrix("ex1data1.txt")
    part3(mat)

    //    val mat1 = 2.0 * Matrix.ones(2,5)
    //    val mat2 = 3.0 * Matrix.ones(5,3)
    //    println("shape(X), shape(theta) = " + shape(mat1) + " " + shape(mat2))
    //    val mat3 = mat1 :* mat2 :* mat2.t
    //
    //    println(mat3)

  }
  main()

}

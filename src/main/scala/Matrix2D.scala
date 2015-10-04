
import cern.colt.function.{DoubleDoubleFunction, DoubleFunction}
import cern.colt.matrix.DoubleFactory2D
import cern.colt.matrix.DoubleMatrix2D
import cern.colt.matrix.impl.DenseDoubleMatrix2D
import cern.colt.matrix.linalg.{Algebra, Property, SingularValueDecomposition}
import cern.jet.math.Functions

import scala.language.implicitConversions

object Matrix2D {


  def fromData[T : Numeric](it : Iterable[Iterable[T]]): Matrix2D = {
    val num = implicitly[Numeric[T]]
    val ar = it.map{ _.map{el => num.toDouble(el)}.toArray }.toArray
    new DenseDoubleMatrix2D(ar)
  }

  def eye(n : Int) = {
    val cMat = new DenseDoubleMatrix2D(n, n)
    for (i <- 0 until n) {
      cMat.setQuick(i, i, 1.0)
    }
    cMat
  }

  def zeros(n : Int, m : Int): Matrix2D = new DenseDoubleMatrix2D(n, m)

  def ones(n : Int, m : Int): Matrix2D = new DenseDoubleMatrix2D(n, m).assign(1.0)

  private def algebra = Algebra.DEFAULT
  private def property = Property.DEFAULT

  // Stack matrices vertically, same as Matrix.appendRows
  // Named for the numpy function
  def vstack(allMats : Matrix2D*) : Matrix2D = {
    DoubleFactory2D.dense.compose(allMats.map { mat => Array(mat.getMat) }.toArray)
  }
  def hstack(allMats : Matrix2D*) : Matrix2D = {
    DoubleFactory2D.dense.compose(Array(allMats.map { _.getMat }.toArray))
  }

  implicit def convertFrom2D(data: DoubleMatrix2D): Matrix2D = {
    new Matrix2D(data)
  }

}

class Matrix2D(val getMat: DoubleMatrix2D) {
  // call-by-name is executed each time, we don't want to do that, so access here:

  val rows = getMat.rows

  val cols = getMat.columns

  val shape = (rows, cols)

  def toArray = getMat.toArray

  // These are expensive so make them lazy
  lazy val det = Matrix2D.algebra.det(getMat)

  // L1 Norm (max abs(sum(col)))
  lazy val norm1 = Matrix2D.algebra.norm1(getMat)
  // L2 Norm, does SVD, max singular value
  lazy val norm2 = Matrix2D.algebra.norm2(getMat)
  // Frobenius Norm = (M * M.t).trace
  lazy val normF = Matrix2D.algebra.normF(getMat)
  // This is the max abs(sum(row))
  lazy val normInf = Matrix2D.algebra.normInfinity(getMat)

  lazy val sum = {
    getMat.zSum()
  }

  lazy val trace = Matrix2D.algebra.trace(getMat)

  lazy val rank = if(isRectangular) {
    Matrix2D.algebra.rank(getMat)
  } else {
    // .viewDice is a view of the transpose
    Matrix2D.algebra.rank(getMat.viewDice)
  }

  lazy val svd = {
    val out = new SingularValueDecomposition(getMat)
    (out.getU, out.getS, out.getV)
  }

  lazy val isRectangular = {
    try {
      Property.DEFAULT.checkRectangular(getMat)
      true
    } catch {
      case(e : IllegalArgumentException) => false
    }
  }

  // square cannot be changed by mapping:
  lazy val isSquare = Property.DEFAULT.isSquare(getMat)

  // After mapping, we may be symmetric, even if we weren't originally:
  lazy val isSymmetric = Property.DEFAULT.isSymmetric(getMat)

  // Transpose
  lazy val T: Matrix2D = getMat.viewDice

  def apply(row: Int, col : Int): Double = {
    getMat.get(row, col)
  }

  def apply(row: Int): Matrix2D = {
    this.getRow(row)
  }

  def update(row: Int, other: Matrix2D): Unit = {
    val that = other.getMat
    for (j <- 0 until cols) {
      getMat.setQuick(row, j, that.getQuick(0, j))
    }
  }

  def apply(x: AnyRef, colSlice : IndexedSeq[Int]): Matrix2D = {
    val width = colSlice.last-colSlice.head
    getMat.viewPart(0, colSlice.head, rows, width)
  }

  def *[T : Numeric](c : T): Matrix2D = {
    val dc = implicitly[Numeric[T]].toDouble(c)
    this.getMat.copy.assign(new DoubleFunction { override def apply(x : Double) = x * dc })
  }

  def *(other : Matrix2D): Matrix2D = {
    this.getMat.zMult(other.getMat, null)
  }

  def +(that : Matrix2D): Matrix2D = {
    this.getMat.copy.assign(that.getMat, Functions.plus)
  }

  def -(that : Matrix2D): Matrix2D = {
    this.getMat.copy.assign(that.getMat, Functions.minus)
  }

  def ==(that: Matrix2D) = {
    Property.DEFAULT.equals(this.getMat, that.getMat)
  }

  def ^(power : Int) : Matrix2D =
    Matrix2D.algebra.pow(getMat, power)

  def getCol(col : Int) = getMat.viewPart(0, col, rows, 1)
  def getRow(row : Int) = getMat.viewPart(row, 0, 1, cols)

  override lazy val hashCode = getMat.hashCode

  override def toString = getMat.toString

}


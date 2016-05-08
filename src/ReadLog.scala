import net.ripe.hadoop.pcap.io.reader.PcapRecordReader
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

import java.nio.file.Paths
import org.joda.time._

import net.ripe.hadoop.pcap.io.PcapInputFormat
import net.ripe.hadoop.pcap.packet.Packet
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark._

import org.apache.hadoop.io.{LongWritable, ObjectWritable}
import org.apache.hadoop.mapred.FileInputFormat
import org.apache.hadoop.mapred.JobConf

//import org.elasticsearch.spark._
import org.joda.time.format.DateTimeFormat

import scala.collection.mutable.ArrayBuffer
//import scalax.io._

/**
  * @author Dhruv
  * @version 1.0
  * Date 5/1/16
  *
  *          This is a implementation class for
  */
object ReadLog {
  val INP_FILE2 = "Data/snort_small.csv" // short data
  val INP_FILE = "Data/myoutput3.txt" // verbose data
//  val OUT_DIR = "/Users/dhruv/Documents/Bigdata/"

  def main (args: Array[String]): Unit =
  {
    val conf = new SparkConf().setAppName("ReadLog").setMaster("local")
    val sc = new SparkContext(conf)

    // Reading the SnortData Logs
    val file_header = sc.textFile(INP_FILE)
    val header = file_header.first();
    // this is file without header
    val file = file_header.filter(s => s!=header)


    val fileLine = file.map(s => (s.split(":")) )
//    val source_ip = fileLine.map( s => s(2) )
//    val dest_ip = fileLine.map( s => s(3))

    println("Time :")

    // Will print time
//    for(column <- fileLine.take(5)) println(column(3).replace("\"","").concat(":").concat(column(4).concat(":").concat(column(5)) )) // Used to test if the file was read correctly

    println(fileLine)
    var sttt:String = null


    for(column <- fileLine.take(5)){

       val len = column(1).split(" ")(1)

      sttt = sttt+" "+ len

//      println(len)
//      println(column(1).split(" ")(1))

    }

    println(sttt)

//    val file_header2 = sc.textFile(INP_FILE2)
//    val fileLine2 = file.map(s => (s.split(",")) )
//
//    val source_ip = file_header2.map( s => s(3) )
//    val dest_ip = fileLine2.map( s => s(3))
//
//    println("Original data short")
//    file_header2.collect().take(5).foreach { println }
//
//    println("Src ")
//    for(column <- fileLine2.take(5)) println(column(2))
  }
}
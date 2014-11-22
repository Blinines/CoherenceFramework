package nlp.framework.discourse;

import java.util.List;
import java.util.Map;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;


/** 
 * Converts text with document markup into parse trees, using Stanford CoreNLP.
 *   
 * @author Karin
 *
 */
public class ParseTreeConverter {
	
	protected StanfordCoreNLP pipeline;
	
	/**
	 * 
	 * @param args filename of file to be parsed. this contains documents with doc tag boundaries
	 * @param args filename of output file for parse trees to be written to. 
	 * This will write to one outputfile with parse trees for each document seperated by doc tags 
	 */
	public static void main(String [] args){
		
		ParseTreeConverter converter = new ParseTreeConverter(); 
		
		converter.parse();
	}
	
	/**
	 * Parses documents into parse trees.
	 * Input and output from console
	 */
	public void parse(){
		
		this.pipeline = new StanfordCoreNLP(new Properties());
		
		Map<String, String> docAndIds = new CorpusReader().readXMLfromConsole();
		
		for(String id : docAndIds.keySet()){
			
			printParseTree(docAndIds.get(id), id);
		}
	}
	
	/**
	 * For each document:
	 * <li>		Read in source text and construct parse tree. 
	 * <li>		 
	 * @param id 
	 *  @return parse tree
	 * @param 
	 */
	public void printParseTree(String docAsString, String id){
		
		
		Annotation document = new Annotation(docAsString);
		this.pipeline.annotate(document);
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);		
		
		
		printStartTag(id);
		for(CoreMap sentence: sentences) {
			
			// this is the parse tree of the current sentence
			Tree root = sentence.get(TreeAnnotation.class);
			if(root.isEmpty()==false){
				System.out.println(root);
			}
			
		}
	}

	
	private void printStartTag(String id) {
		System.out.println("\n # id="+id+"\n");
		
	}

	/**
	 * Parses documents into parse trees.
	 * @param filename of file to be parsed. this contains documents with doc tag boundaries
	 * @param  outputfile, filename of output file for parse trees to be written to. 
	 * This will write to one outputfile with parse trees for each document separated by doc tags 
	 */
	public void parse(String filename, String outputfile){
		
		this.pipeline = new StanfordCoreNLP(new Properties());
		
		List<String> docs = new CorpusReader().readXML(filename, true);
		
		StringBuffer trees = new StringBuffer();
				
		for(String docAsString: docs){
			
			getParseTree(docAsString, trees);
		}
		
		FileOutputUtils.streamToFile(outputfile, trees);
	}
	
	/**
	 * For each document:
	 * <li>		Read in source text and construct parse tree. 
	 * <li>		 
	 *  @return parse tree
	 * @param 
	 */
	public StringBuffer getParseTree(String docAsString, StringBuffer trees){
		
		Annotation document = new Annotation(docAsString);
		this.pipeline.annotate(document);
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);		
		
		for(CoreMap sentence: sentences) {
			
			// this is the parse tree of the current sentence
			Tree root = sentence.get(TreeAnnotation.class);
			
			trees.append(root);
			trees.append('\n');
			
		}
		
		
		return trees;
	}


}

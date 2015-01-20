package nlp.framework.discourse;

import java.util.List;

import junit.framework.TestCase;

public class BipartiteGraphTest extends TestCase {
	
	public static final String teststring1 = "I am going to travel to Berlin. Berlin is a very cosmopolitan city in Germany. The city is just buzzing.";
	public static final String teststring2 = "The atom is a basic unit of matter, it consists of a dense central nucleus surrounded by a cloud of negatively charged electrons.";
	public static final String incoherentString1 = "The atom is a basic unit of matter, it consists of a dense central nucleus surrounded by a cloud of negatively charged electrons. The capital of France is Paris. Bananas are a type of fruit.";
	public static final String message = "Entity incorrectly extracted.";
	public static final String message2 = "Coherence incorrectly computed.";
	public static final String messageXml = "Wrong number of docs extracted from xml.";
	public static final String xml = "<refset setid=\"newsdev2009\"><doc docid=\"napi.hu/2007/12/12/0\" genre=\"news\"><hl><seg id=\"1\"> Food: Where European inflation slipped up </seg>"+
			"</hl><p><seg id=\"2\"> The skyward zoom in food prices is the dominant force behind the speed up in eurozone inflation. </seg></p></doc></refset>";
	public static final String xml2 ="<refset setid=\"newsdev2009\"><doc docid=\"napi.hu/2007\"><hl><seg id=\"1\"> Government crisis coming, says Gallup. </seg></hl><p><seg id=\"2\">"+
			"Fidesz support shot up significantly in early December after a lengthy period of just holding its own. "+
			"This gives it the strongest support base it has seen since 2002, while Gallup reports support for the socialists at an all-time low of 13 percent. </seg></p></doc>"+
			"<doc><p><seg id=\"15\"> In comparison, only 55 percent said it would definitely vote in parliamentary elections if they were to be held this Sunday while another 15 percent said it most likely would cast a ballot. "
			+"Given current determination to vote, the referendum is certain to be valid. </seg></p></doc></refset>";

	public static final String xmlFrench = "<refset setid=\"newsdev2009\"><doc docid=\"napi.hu/2007/12/12/0\" genre=\"news\"><hl><seg id=\"1\"> L'inflation, en Europe, a d�rap� sur l'alimentation. </seg>"+
			"</hl><p><seg id=\"2\">  L'inflation acc�l�r�e, mesur�e dans la zone euro, est due principalement � l'augmentation rapide des prix de l'alimentation. </seg></p></doc></refset>";
	  
	public static final String xmlFrench2 ="<refset setid=\"newsdev2009\"><doc docid=\"napi.hu/2007\"><hl><seg id=\"1\">Gallup indique une crise du gouvernement. </seg></hl><p><seg id=\"2\">"+
			"Apr�s une longue stagnation, le nombre des partisans du Fidesz (Alliance des jeunes d�mocrates) a augment� significativement d�but d�cembre. "+
			"Ainsi depuis 2002, il dispose actuellement du plus grand nombre de partisans, alors que Gallup (Institut de sondage) n'a jamais constat� auparavant un aussi faible soutien socialiste, � savoir 13 pour cent. </seg></p></doc>"+
			"<doc><p><seg id=\"15\"> Pour faire la comparaison, pour une �lection l�gislative organiser dimanche prochain, seulement 55 pour cent des sond�s participeraient certainement et 15 pour cent compl�mentaires participeraient probablement. "
			+"Suivant les intentions actuelles, le r�sultat du r�f�rendum ne ferait aucun doute. </seg></p></doc></refset>";

	
	@Override
	protected void setUp() throws Exception {		
		super.setUp();
	}

	
	public void testConstructSentenceAndEntityNodes(){
		
		EntityGraph graph = new EntityGraph("default", new EntityGridFramework());
		//String doc = CorpusReader.readDataAsString(filename);
		BipartiteGraph bipartiteGraph  = graph.identifyEntitiesAndConstructGraph(teststring1);
		assertEquals(message, 3, bipartiteGraph.getSentenceNodes().size());
		assertEquals(message, 3, bipartiteGraph.getEntities().size());
		
		assertEquals(message, true, bipartiteGraph.containsSentenceNode(new Integer(2)) != null);
		
		assertEquals(message, true, bipartiteGraph.containsEntityNode(("Berlin")));
		
		assertEquals(message, true, bipartiteGraph.containsSentenceNode(new Integer(1)).hasEntityNode("Berlin"));
		
		assertEquals(message, true, bipartiteGraph.containsSentenceNode(new Integer(2)).hasEntityNode("Berlin") == false);
		
		assertEquals(message, true, bipartiteGraph.containsSentenceNode(new Integer(1)).hasEdge("Berlin",  BipartiteGraph.SUBJECT));
		assertEquals(message, true, bipartiteGraph.containsSentenceNode(new Integer(1)).hasEdge("city",  BipartiteGraph.OTHER_GRAMMATICAL));
		
		assertEquals(message, true, bipartiteGraph.containsSentenceNode(new Integer(0)).hasEdge("Berlin",  BipartiteGraph.OBJECT));
		
		assertEquals(message, true, bipartiteGraph.containsSentenceNode(new Integer(2)).hasEdge("city",  BipartiteGraph.SUBJECT));
		
		assertEquals(message, 1, bipartiteGraph.containsSentenceNode(new Integer(2)).getAllEdges().size());
		
	}
	
	/**
	 * check that a document is correctly extracted from an xml segment 
	 */
	/*public void testXmlExtractDocs1(){
		EntityGraph graph = new EntityGraph(fileName, language);
		List<String> docs = new CorpusReader().readXMLString(xml);
		int fileidx = 0;
		for(String docAsString: docs){
			
			char grid [][] = graph..identifyEntitiesAndConstructGrid(docAsString);
			//FileOutputUtils.writeGridToFile(outputfile+fileidx, grid);
			fileidx++;
		}
		assertEquals(messageXml, 1, fileidx);
	}*/
	/**
	 * check that a document is correctly extracted from an xml segment and that resulting graph has correct structure
	 */
	public void testFrenchGraphCorrectStructure(){
		
		EntityGraph graph = new EntityGraph("French",  new EntityGridFactory().getEntityGridFramework( "French", EntityGridFramework.FRENCH_TAGGER));
		List<String> docs = new CorpusReader().readXMLString(xmlFrench);
		BipartiteGraph bipartiteGraph  = graph.identifyEntitiesAndConstructGraph(docs.get(0));
		/*int fileidx = 0;
		for(String docAsString: docs){
			
			BipartiteGraph bipartiteGraph  = graph.ientifyEntitiesAndConstructGraph(docAsString);
			//FileOutputUtils.writeGridToFile(outputfile+fileidx, grid);
			fileidx++;
		}
		assertEquals(messageXml, 1, fileidx);*/

		assertEquals(message, 2, bipartiteGraph.getSentenceNodes().size());
		assertEquals(message, 8, bipartiteGraph.getEntities().size());
		
		assertEquals(message, bipartiteGraph.containsSentenceNode(new Integer(2)), true);
		
		assertEquals(message, bipartiteGraph.containsEntityNode(("partisans")));
		
		//2 entities shared over 2 sentences: 6/2=3 
		assertEquals(message, 3, bipartiteGraph.getLocalCoherence(BipartiteGraph.UNWEIGHTED_PROJECTION));

		assertEquals(message, 3, bipartiteGraph.getLocalCoherence( BipartiteGraph.UNWEIGHTED_PROJECTION));
 
	}
	
	/**
	 * Tests whether graph is correctly computing local coherence value
	 */
	public void testCoherenceValue(){
		EntityGraph graph = new EntityGraph("default", new EntityGridFramework());
		//String doc = CorpusReader.readDataAsString(filename);
		BipartiteGraph bipartiteGraph  = graph.identifyEntitiesAndConstructGraph(teststring1);
		assertEquals(message, 3, bipartiteGraph.getSentenceNodes().size());
		assertEquals(message, 3, bipartiteGraph.getEntities().size());
		SentenceNode sentence = bipartiteGraph.getSentenceNodes().iterator().next();
		assertEquals(message, true, sentence.hasEntityNode("Berlin"));
		assertEquals(message, true,bipartiteGraph.containsSentenceNode(new Integer(2)) != null);
		
		assertEquals(message, true, bipartiteGraph.containsEntityNode(("Berlin")));
		
		//3 sentences, 2 shared entities :1.0/3.0 * 2.0
		assertEquals(message, Math.round(1.0/3.0 * 2.0) , Math.round(bipartiteGraph.getLocalCoherence(BipartiteGraph.UNWEIGHTED_PROJECTION)));
		
 	}
	
	/**
	 * Tests whether graph is correctly computing local coherence value
	 */
	public void testCoherenceValueRanking(){
		EntityGraph graph = new EntityGraph("default", new EntityGridFramework());
		//String doc = CorpusReader.readDataAsString(filename);
		BipartiteGraph bipartiteGraph  = graph.identifyEntitiesAndConstructGraph(incoherentString1);
		assertEquals(message, 3, bipartiteGraph.getSentenceNodes().size());
		assertEquals(message, 12, bipartiteGraph.getEntities().size());
		SentenceNode sentence = bipartiteGraph.containsSentenceNode(0);
		assertEquals(message, true, sentence.hasEntityNode("atom"));
		assertEquals(message, true,bipartiteGraph.containsSentenceNode(new Integer(2)) != null);
		assertEquals(message, true, bipartiteGraph.containsEntityNode(("France")));
		
		BipartiteGraph bipartiteGraph2  = graph.identifyEntitiesAndConstructGraph(teststring1);
		
		//3 sentences, 2 shared entities :1/3 * 5/3
		assertTrue(message2,  Math.round(bipartiteGraph2.getLocalCoherence(	BipartiteGraph.UNWEIGHTED_PROJECTION)) >  Math.round(bipartiteGraph.getLocalCoherence( BipartiteGraph.UNWEIGHTED_PROJECTION)));
		
 	}
	
	
	/**
	 * check that a document is correctly extracted from an xml segment 
	 */
	/*public void testXmlExtractDocs2(){
		EntityGraph graph = new EntityGraph(fileName, language);
		List<String> docs = new CorpusReader().readXMLString(xml2);
		int fileidx = 0;
		for(String docAsString: docs){
			
			char grid [][] = gridframework.identifyEntitiesAndConstructGrid(docAsString);
			//FileOutputUtils.writeGridToFile(outputfile+fileidx, grid);
			fileidx++;
		}
		assertEquals(messageXml, 2, fileidx);
	}*/	


}

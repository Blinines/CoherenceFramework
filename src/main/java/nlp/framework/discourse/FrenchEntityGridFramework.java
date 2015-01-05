package nlp.framework.discourse;

import java.util.Properties;

import edu.stanford.nlp.pipeline.StanfordCoreNLP;


/**
 * Extends the parent class to add functionality pertinent to a model for the German language.
 * @author Karin Sim
 *
 */
public class FrenchEntityGridFramework  extends EntityGridFramework {

	public FrenchEntityGridFramework(String url) {
		super();
		Properties properties = new Properties();	
		properties.put("-parseInside", "HEADLINE|P");
		properties.put("annotators", "tokenize, ssplit, pos, lemma, parse");
		properties.put("parse.flags", "");
		properties.put("parse.model", "edu/stanford/nlp/models/lexparser/frenchFactored.ser.gz");
		properties.put("pos.model",url);	
		this.pipeline = new StanfordCoreNLP(properties);
		System.out.println("FrenchEntityGrid- setting to "+url);
	}
}

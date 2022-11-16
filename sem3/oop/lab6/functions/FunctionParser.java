package functions;

import java.text.ParseException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import functions.basic.Const;
import functions.basic.Cos;
import functions.basic.Exp;
import functions.basic.Identity;
import functions.basic.Log;
import functions.basic.Sin;

public class FunctionParser {
	static final Pattern pattern = Pattern
			.compile("(cos|sin|exp|ln)|([+\\-*^])|([x])|([\\(\\)])|(([0-9]*[.]?[0-9]+|[0-9]+[.]?[0-9]*))|(\\S+)");

	public static Function parse(String query) throws ParseException {
		Matcher matcher = pattern.matcher(query);
		Function result = nextToken(matcher);
		if (result == null)
			throw new ParseException("Invalid format!", matcher.start());
		return result;
	}

	private static Function nextToken(Matcher matcher) throws ParseException {
		Function result = null;
		while (matcher.find()) {
			String curToken = matcher.group();
			switch (curToken) {
				case "(":
					result = nextToken(matcher);
					break;
				case ")":
					if (result == null)
						throw new ParseException("Invalid format!", matcher.start());
					return result;
				case "cos":
					result = Functions.composition(new Cos(), nextToken(matcher));
					break;
				case "sin":
					result = Functions.composition(new Sin(), nextToken(matcher));
					break;
				case "ln":
					result = Functions.composition(new Log(), nextToken(matcher));
					break;
				case "exp":
					result = Functions.composition(new Exp(), nextToken(matcher));
					break;
				case "+":
					result = Functions.sum(result, nextToken(matcher));
					break;
				case "-":
					result = Functions.sum(result, Functions.mult(nextToken(matcher), new Const(-1)));
					break;
				case "^":
					result = Functions.power(result, nextToken(matcher));
					break;
				case "*":
					result = Functions.mult(result, nextToken(matcher));
					break;
				case "x":
					result = new Identity();
					break;
				default:
					try {
						return new Const(Double.parseDouble(curToken));
					} catch (NumberFormatException e) {
						throw new ParseException("Unexcepted token!", matcher.start());
					}
			}
		}
		if (result == null)
			throw new ParseException("Invalid format!", matcher.start());
		return result;
	}

}

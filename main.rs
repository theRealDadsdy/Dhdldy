use crate::compiler::compile;
use crate::parser::full_parse;

mod tokenizer {
    use std::iter::Peekable;
    use std::mem::replace;
    use std::rc::Rc;
    use std::str::Chars;

    #[derive(Debug, Clone, Copy)]
    pub enum Keyword {
        FuncDef,
        TypeDef,
        Error,
        Map,
        Foreach,
        If,
        Else,
        Input,
        Output,
    }

    fn get_keyword(string: &str) -> Option<Keyword> {
        match string {
            "comp" => Some(Keyword::FuncDef),
            "newtype" => Some(Keyword::TypeDef),
            "error" => Some(Keyword::Error),
            "map" => Some(Keyword::Map),
            "foreach" => Some(Keyword::Foreach),
            "if" => Some(Keyword::If),
            "else" => Some(Keyword::Else),
            "INPUT" => Some(Keyword::Input),
            "OUTPUT" => Some(Keyword::Output),
            _ => None,
        }
    }

    #[derive(Debug, Clone)]
    pub enum Token {
        OpeningParen,
        ClosingParen,
        OpeningBrace,
        ClosingBrace,
        OpeningBracket,
        ClosingBracket,
        OpeningAngle,
        ClosingAngle,
        Spread,
        Bar,
        Comma,
        Semicolon,
        Colon,
        Assoc,
        Arrow,
        Equals,
        At,
        Dollar,
        Exclamation,
        Numeral(u128),
        Char(char),
        Keyword(Keyword),
        Ident(Rc<str>),
        String(String),
    }

    impl std::fmt::Display for Keyword {
        fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                Keyword::FuncDef => write!(formatter, "comp"),
                Keyword::TypeDef => write!(formatter, "typedef"),
                Keyword::Error => write!(formatter, "error"),
                Keyword::Map => write!(formatter, "map"),
                Keyword::Foreach => write!(formatter, "foreach"),
                Keyword::If => write!(formatter, "if"),
                Keyword::Else => write!(formatter, "else"),
                Keyword::Input => write!(formatter, "INPUT"),
                Keyword::Output => write!(formatter, "OUTPUT"),
            }
        }
    }

    impl std::fmt::Display for Token {
        fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                Token::OpeningParen => write!(formatter, "("),
                Token::ClosingParen => write!(formatter, ")"),
                Token::OpeningBrace => write!(formatter, "{{"),
                Token::ClosingBrace => write!(formatter, "}}"),
                Token::OpeningBracket => write!(formatter, "["),
                Token::ClosingBracket => write!(formatter, "]"),
                Token::OpeningAngle => write!(formatter, "<"),
                Token::ClosingAngle => write!(formatter, ">"),
                Token::Spread => write!(formatter, ".."),
                Token::Bar => write!(formatter, "|"),
                Token::Comma => write!(formatter, ","),
                Token::Colon => write!(formatter, ":"),
                Token::Assoc => write!(formatter, "::"),
                Token::Semicolon => write!(formatter, ";"),
                Token::Arrow => write!(formatter, "->"),
                Token::Equals => write!(formatter, "="),
                Token::At => write!(formatter, "@"),
                Token::Dollar => write!(formatter, "$"),
                Token::Exclamation => write!(formatter, "!"),
                Token::Numeral(num) => write!(formatter, "{num}"),
                Token::Char(ch) => write!(formatter, "'{ch}"),
                Token::Keyword(keyword) => write!(formatter, "{keyword}"),
                Token::Ident(name) => write!(formatter, "{name}"),
                Token::String(err) => write!(formatter, "\"{err}\""),
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub enum TokenError {
        InvalidChar(char),
        EoF,
    }

    impl std::fmt::Display for TokenError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                TokenError::InvalidChar(char) => write!(formatter, "Invalid char `{}`", char),
                TokenError::EoF => write!(formatter, "Unexpected end of input"),
            }
        }
    }

    pub struct TokenStream<'a>(
        Peekable<Chars<'a>>,
        Option<Result<Token, TokenError>>,
        Option<Result<Token, TokenError>>,
    );

    impl TokenStream<'_> {
        pub fn new(string: &str) -> TokenStream {
            TokenStream(string.chars().peekable(), None, None)
        }
        pub fn peek(&mut self) -> Result<Token, TokenError> {
            let res = self.token();
            self.2 = replace(&mut self.1, Some(res.clone()));
            res
        }
        pub fn peek2(&mut self) -> Result<Token, TokenError> {
            let res1 = self.token();
            let res2 = self.token();
            self.1 = Some(res1);
            self.2 = Some(res2.clone());
            res2
        }
        pub fn token(&mut self) -> Result<Token, TokenError> {
            if self.1.is_some() {
                return replace(&mut self.1, self.2.take()).expect("It was `some`");
            }
            loop {
                let curr = self.0.next().ok_or(TokenError::EoF)?;
                return Ok(match curr {
                    '(' => Token::OpeningParen,
                    ')' => Token::ClosingParen,
                    '{' => Token::OpeningBrace,
                    '}' => Token::ClosingBrace,
                    '[' => Token::OpeningBracket,
                    ']' => Token::ClosingBracket,
                    '<' => Token::OpeningAngle,
                    '>' => Token::ClosingAngle,
                    '|' => Token::Bar,
                    ',' => Token::Comma,
                    ';' => Token::Semicolon,
                    ':' => match self.0.peek() {
                        Some(':') => {
                            self.0.next();
                            Token::Assoc
                        }
                        _ => Token::Colon,
                    },
                    '=' => Token::Equals,
                    '@' => Token::At,
                    '$' => Token::Dollar,
                    '!' => Token::Exclamation,
                    '-' => match self.0.next() {
                        Some('>') => Token::Arrow,
                        Some(_) => Err(TokenError::InvalidChar('-'))?,
                        None => Err(TokenError::EoF)?,
                    },
                    '.' => match self.0.next() {
                        Some('.') => Token::Spread,
                        Some(_) => Err(TokenError::InvalidChar('.'))?,
                        None => Err(TokenError::EoF)?,
                    },
                    '\'' => Token::Char(self.0.next().ok_or(TokenError::EoF)?),
                    '"' => {
                        let mut string: String = "".into();
                        loop {
                            match self.0.next().ok_or(TokenError::EoF)? {
                                '"' => break,
                                a => string.push(a),
                            }
                        }
                        Token::String(string)
                    }
                    '0'..='9' => {
                        let mut num = curr as u128 - 48;
                        loop {
                            match self.0.peek() {
                                Some('0'..='9') => {
                                    num = 10 * num + self.0.next().ok_or(TokenError::EoF)? as u128
                                        - 48
                                }
                                _ => return Ok(Token::Numeral(num)),
                            }
                        }
                    }
                    'a'..='z' | 'A'..='Z' | '_' => {
                        let mut name = curr.to_string();
                        loop {
                            match self.0.peek() {
                                Some('a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '.') => {
                                    name.push(self.0.next().ok_or(TokenError::EoF)?)
                                }
                                _ => {
                                    if let Some(keyword) = get_keyword(&name) {
                                        return Ok(Token::Keyword(keyword));
                                    } else {
                                        return Ok(Token::Ident(name.into()));
                                    }
                                }
                            }
                        }
                    }
                    '/' => match self.0.next() {
                        Some('/') => {
                            loop {
                                match self.0.next() {
                                    Some('\n') => break,
                                    Some(_) => (),
                                    None => Err(TokenError::EoF)?,
                                }
                            }
                            continue;
                        }
                        Some(_) => Err(TokenError::InvalidChar('/'))?,
                        None => Err(TokenError::EoF)?,
                    },
                    ' ' | '\n' | '\t' => continue,
                    c => Err(TokenError::InvalidChar(c))?,
                });
            }
        }
    }
}
mod parser {
    use crate::tokenizer::{Keyword, Token, TokenError, TokenStream};
    use std::iter;
    use std::rc::Rc;

    macro_rules! pat_names {
        ( $( $pat: pat ),* ) => {
            vec!($(stringify!($pat)),*)
        }
    }
    macro_rules! match_token {
        ( $tokens: ident, $( $pat: pat => $expr: expr),* $(,)? ) => {
            match $tokens.token()?{
                $(
                    $pat => $expr,
                )*
                token => return Err(ParseError::UnexpectedToken(token, pat_names!($($pat),*), line!())),
            }
        };
    }
    macro_rules! assert_token {
        ( $tokens: ident, $( $pat: pat ),* ) => {
            match $tokens.token()?{
                $(
                    $pat => (),
                )*
                token => return Err(ParseError::UnexpectedToken(token, pat_names!($($pat),*), line!())),
            }
        };
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum Statement {
        Rule(Rule),
        Foreach(Foreach),
        If(If),
    }
    #[derive(Debug, Clone, PartialEq)]
    pub struct If(pub Expression, pub Vec<Statement>);
    #[derive(Debug, Clone, PartialEq)]
    pub struct Foreach(pub Rc<str>, pub Expression, pub Vec<Statement>);
    #[derive(Debug, Clone, PartialEq)]
    pub struct Rule(pub Rc<str>, pub Vec<Expression>, pub bool, pub Expression);
    #[derive(Debug, Clone, PartialEq)]
    pub enum Expression {
        String(String),
        Error(String),
        Select(Box<Expression>, Box<Expression>, Box<Expression>),
        Inc(Box<Expression>),
        Dec(Box<Expression>),
        IsZero(Box<Expression>),
        Map(Rc<str>, Box<Expression>, Box<Expression>),
        VarAccess(Rc<str>),
        TypedVarAccess(Rc<str>, Box<Expression>),
        Circular(Box<Expression>),
        Typedef(Box<Expression>),
        Wrap(usize, Box<Expression>),
        Unwrap(Box<Expression>),
        Tuple(Vec<Item<Expression>>),
        Array(Box<Expression>, Box<Expression>),
        Index(Box<Expression>, Index),
        Call(Box<Expression>, Vec<Item<Expression>>),
        Block(Vec<Statement>, Box<Expression>),
        Component(
            Vec<Item<(Rc<str>, Expression)>>,
            Vec<Rc<str>>,
            Box<Expression>,
            Box<Expression>,
        ),
        CompType(Vec<Expression>, Box<Expression>),
        Number(u128, Option<u32>),
    }
    #[derive(Debug, Clone, PartialEq)]
    pub enum Item<T> {
        Value(T),
        Spread(T),
    }
    impl<T> Item<T> {
        pub fn unwrap(self) -> T {
            match self {
                Item::Value(val) => val,
                Item::Spread(val) => val,
            }
        }
    }
    #[derive(Debug, Clone, PartialEq)]
    pub enum Index {
        Number(Box<Expression>),
        Range(Box<Expression>, Box<Expression>),
    }
    pub enum ParseError {
        TokenError(TokenError),
        UnexpectedToken(Token, Vec<&'static str>, u32),
        InvalidNumeral(u128),
    }

    impl From<TokenError> for ParseError {
        fn from(token_err: TokenError) -> ParseError {
            ParseError::TokenError(token_err)
        }
    }

    impl std::fmt::Display for ParseError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                ParseError::TokenError(err) => write!(formatter, "{err}"),
                ParseError::UnexpectedToken(token, expected, line) => {
                    writeln!(formatter, "Unexpected token: `{token}` (line {line})")?;
                    write!(formatter, "Expected ")?;
                    for expect in expected {
                        write!(formatter, "`{expect}`")?;
                    }
                    Ok(())
                }
                ParseError::InvalidNumeral(num) => {
                    write!(formatter, "Invalid numeric value literal `{num}`. Try annotating it's size like `{num}u128`;")
                }
            }
        }
    }

    fn try_parse_statement(tokens: &mut TokenStream) -> Option<Result<Statement, ParseError>> {
        match (tokens.peek(), tokens.peek2()) {
            (Ok(Token::Keyword(Keyword::Error)), _) => Some({
                (|| {
                    let error = parse_expression(tokens)?;
                    assert_token!(tokens, Token::Semicolon);
                    Ok(Statement::Rule(Rule("*".into(), vec![], false, error)))
                })()
            }),
            (Ok(Token::Keyword(Keyword::TypeDef)), _) => {
                Some(parse_typedef(tokens).map(Statement::Rule))
            }
            (Ok(Token::Ident(_)), Ok(Token::Colon | Token::Equals)) => {
                Some(parse_rule(tokens).map(Statement::Rule))
            }
            (Ok(Token::Keyword(Keyword::FuncDef)), _) => {
                Some(parse_func_def(tokens).map(Statement::Rule))
            }
            (Ok(Token::Keyword(Keyword::Foreach)), _) => {
                Some(parse_foreach(tokens).map(Statement::Foreach))
            }
            (Ok(Token::Keyword(Keyword::If)), _) => Some(parse_if(tokens).map(Statement::If)),
            _ => None,
        }
    }

    fn parse_if(tokens: &mut TokenStream) -> Result<If, ParseError> {
        assert_token!(tokens, Token::Keyword(Keyword::If));
        let expr = parse_expression(tokens)?;
        assert_token!(tokens, Token::OpeningBrace);
        let rules = parse_statements(tokens)?;
        assert_token!(tokens, Token::ClosingBrace);
        Ok(If(expr, rules))
    }

    fn parse_foreach(tokens: &mut TokenStream) -> Result<Foreach, ParseError> {
        assert_token!(tokens, Token::Keyword(Keyword::Foreach));
        let name = match_token!(tokens, Token::Ident(name) => name);
        assert_token!(tokens, Token::Colon);
        let expr = parse_expression(tokens)?;
        assert_token!(tokens, Token::OpeningBrace);
        let rules = parse_statements(tokens)?;
        assert_token!(tokens, Token::ClosingBrace);
        Ok(Foreach(name, expr, rules))
    }

    fn parse_typedef(tokens: &mut TokenStream) -> Result<Rule, ParseError> {
        assert_token!(tokens, Token::Keyword(Keyword::TypeDef));
        let name = match_token!(tokens, Token::Ident(name) => name);
        let expr = Expression::Typedef(Box::new(parse_expression(tokens)?));
        assert_token!(tokens, Token::Semicolon);
        Ok(Rule(name, vec![], false, expr))
    }

    fn parse_func_def(tokens: &mut TokenStream) -> Result<Rule, ParseError> {
        assert_token!(tokens, Token::Keyword(Keyword::FuncDef));
        let name = match_token!(tokens, Token::Ident(name) => name);
        let generics = match_token!(tokens,
            Token::OpeningAngle => {
                let mut generics = vec![];
                loop {
                    match_token!(tokens,
                        Token::Ident(name) => generics.push(name),
                        Token::ClosingAngle => break,
                    )
                }
                assert_token!(tokens, Token::OpeningParen);
                generics
            },
            Token::OpeningParen => vec![],
        );
        let mut args = vec![];
        loop {
            match_token!(tokens,
                Token::Comma => continue,
                Token::ClosingParen => break,
                Token::Ident(name) => {
                    assert_token!(tokens, Token::Colon);
                    args.push(Item::Value((name, parse_expression(tokens)?)));
                }
            )
        }
        assert_token!(tokens, Token::Arrow);
        let ret = parse_expression(tokens)?;
        assert_token!(tokens, Token::OpeningBrace);
        let block = parse_block(tokens)?;
        Ok(Rule(
            name,
            vec![],
            false,
            Expression::Component(args, generics, Box::new(ret), Box::new(block)),
        ))
    }

    fn parse_rule(tokens: &mut TokenStream) -> Result<Rule, ParseError> {
        let name = match_token!(tokens,
            Token::Ident(name) => name,
        );
        let ty = match tokens.peek()? {
            Token::Colon => {
                tokens.token()?;
                Some(parse_expression(tokens)?)
            }
            _ => None,
        };
        assert_token!(tokens, Token::Equals);
        let expr = parse_expression(tokens)?;
        assert_token!(tokens, Token::Semicolon);
        let mut subs = vec![expr.clone()];
        let mut reqs = match ty {
            Some(ty) => vec![ty],
            None => vec![],
        };
        let mut circ = false;
        while let Some(expr) = subs.pop() {
            use crate::parser::Expression as E;
            match expr {
                E::Tuple(exprs) => subs.extend(exprs.into_iter().map(Item::unwrap)),
                E::Call(expr, exprs) => {
                    subs.push(*expr);
                    subs.extend(exprs.into_iter().map(Item::unwrap));
                }
                E::Circular(ty) => {
                    reqs.push(*ty);
                    circ = true;
                }
                E::Index(expr, _) => subs.push(*expr),
                E::Unwrap(expr) => subs.push(*expr),
                E::TypedVarAccess(_, expr) => subs.push(*expr),
                E::Select(expr1, expr2, expr3) => {
                    subs.push(*expr1);
                    subs.push(*expr2);
                    subs.push(*expr3);
                }
                E::Typedef(expr) => subs.push(*expr),
                E::Wrap(_, expr) => subs.push(*expr),
                E::CompType(mut args, expr) => {
                    subs.append(&mut args);
                    subs.push(*expr);
                }
                E::Component(args, _, ret, expr) => {
                    subs.extend(args.into_iter().map(|x| x.unwrap().1));
                    subs.push(*ret);
                    subs.push(*expr);
                }
                E::Map(_, expr, body) => {
                    subs.push(*expr);
                    subs.push(*body);
                }
                E::Array(expr, reps) => {
                    subs.push(*expr);
                    subs.push(*reps);
                }
                E::Inc(expr) => subs.push(*expr),
                E::Dec(expr) => subs.push(*expr),
                E::IsZero(expr) => subs.push(*expr),
                E::Error(_) | E::Number(..) | E::String(_) | E::VarAccess(_) | E::Block(..) => (),
            }
        }
        Ok(Rule(name, reqs, circ, expr))
    }

    fn parse_expression(tokens: &mut TokenStream) -> Result<Expression, ParseError> {
        let mut res = match_token!(tokens,
            Token::Keyword(Keyword::Map) => parse_map(tokens)?,
            Token::String(string) => Expression::String(string),
            Token::Keyword(Keyword::Error) => Expression::Error(match_token!(tokens, Token::String(string)=>string)),
            Token::Keyword(Keyword::FuncDef) => Expression::CompType(
                {
                    assert_token!(tokens, Token::OpeningParen);
                    let mut args = vec![];
                    loop {
                        if let Ok(Token::ClosingParen) = tokens.peek() {
                            let _ = tokens.token();
                            break args;
                        }
                        args.push(parse_expression(tokens)?);
                        match_token!(tokens,
                            Token::Comma => {},
                            Token::ClosingParen => break args,
                        );
                    }
                },
                {
                    assert_token!(tokens, Token::Arrow);
                    Box::new(parse_expression(tokens)?)
                }
            ),
            Token::At => Expression::Circular(Box::new(parse_expression(tokens)?)),
            Token::Numeral(num) => {
                let size = match tokens.peek() {
                    Ok(Token::Ident(string)) => {
                        let mut last = string.chars();
                        let first = last.next().ok_or(TokenError::EoF)?;
                        if first != 'u' {
                            return Err(ParseError::InvalidNumeral(num));
                        }
                        tokens.token()?;
                        let size = last.collect::<String>().parse().map_err(|_| ParseError::InvalidNumeral(num))?;
                        if num.checked_shr(size).unwrap_or(0) > 0 {
                            return Err(ParseError::InvalidNumeral(num));
                        }
                        Some(size)
                    }
                    _ => None,
                };
                Expression::Number(num, size)
            },
            Token::Char(ch) => Expression::Number(ch as u128, Some(8)),
            Token::OpeningBracket => parse_array(tokens)?,
            Token::OpeningParen => parse_tuple(tokens)?,
            Token::OpeningBrace => parse_block(tokens)?,
            Token::Bar => parse_component(tokens)?,
            Token::Ident(name) => match tokens.peek() {
                Ok(Token::Assoc) => {
                    tokens.token()?;
                    assert_token!(tokens, Token::OpeningAngle);
                    let ty = parse_expression(tokens)?;
                    assert_token!(tokens, Token::ClosingAngle);
                    Expression::TypedVarAccess(name, Box::new(ty))
                }
                _ => Expression::VarAccess(name),
            },
        );
        loop {
            res = match tokens.peek() {
                Ok(Token::OpeningBracket) => parse_indexing(res, tokens)?,
                Ok(Token::OpeningParen) => parse_call(res, tokens)?,
                Ok(Token::Exclamation) => {
                    let _ = tokens.token();
                    Expression::Unwrap(Box::new(res))
                }
                _ => return Ok(res),
            }
        }
    }

    fn parse_item(tokens: &mut TokenStream) -> Result<Item<Expression>, ParseError> {
        match tokens.peek() {
            Ok(Token::Spread) => {
                let _ = tokens.token();
                parse_expression(tokens).map(Item::Spread)
            }
            _ => parse_expression(tokens).map(Item::Value),
        }
    }

    fn parse_map(tokens: &mut TokenStream) -> Result<Expression, ParseError> {
        let name = match_token!(tokens, Token::Ident(name) => name);
        assert_token!(tokens, Token::Colon);
        let expr = parse_expression(tokens)?;
        let body = parse_expression(tokens)?;
        Ok(Expression::Map(name, Box::new(expr), Box::new(body)))
    }

    fn parse_array(tokens: &mut TokenStream) -> Result<Expression, ParseError> {
        let mut exprs = vec![];
        loop {
            match tokens.peek() {
                Ok(Token::ClosingBracket) => {
                    tokens.token()?;
                    return Ok(Expression::Tuple(exprs));
                }
                _ => {
                    let expr = parse_item(tokens)?;
                    match_token!(tokens,
                        Token::Comma => exprs.push(expr),
                        Token::Semicolon => {
                            let reps = parse_expression(tokens)?;
                            assert_token!(tokens, Token::ClosingBracket);
                            return Ok(Expression::Array(Box::new(expr.unwrap()), Box::new(reps)));
                        },
                        Token::ClosingBracket => {
                            exprs.push(expr);
                            return Ok(Expression::Tuple(exprs));
                        }
                    )
                }
            }
        }
    }

    fn parse_tuple(tokens: &mut TokenStream) -> Result<Expression, ParseError> {
        let mut exprs = vec![];
        match tokens.peek() {
            Ok(Token::ClosingParen) => return Ok(Expression::Tuple(exprs)),
            _ => {
                let expr = parse_expression(tokens)?;
                match_token!(tokens,
                    Token::Comma => exprs.push(Item::Value(expr)),
                    Token::ClosingParen => return Ok(expr),
                )
            }
        }
        loop {
            match tokens.peek() {
                Ok(Token::ClosingParen) => {
                    tokens.token()?;
                    return Ok(Expression::Tuple(exprs));
                }
                _ => {
                    exprs.push(Item::Value(parse_expression(tokens)?));
                    match_token!(tokens,
                        Token::Comma => (),
                        Token::ClosingParen => return Ok(Expression::Tuple(exprs)),
                    )
                }
            }
        }
    }

    fn parse_call(func: Expression, tokens: &mut TokenStream) -> Result<Expression, ParseError> {
        assert_token!(tokens, Token::OpeningParen);
        let mut exprs = vec![];
        loop {
            match tokens.peek() {
                Ok(Token::ClosingParen) => {
                    tokens.token()?;
                    return Ok(Expression::Call(Box::new(func), exprs));
                }
                _ => {
                    exprs.push(Item::Value(parse_expression(tokens)?));
                    match_token!(tokens,
                        Token::Comma => (),
                        Token::ClosingParen => return Ok(Expression::Call(Box::new(func), exprs)),
                    )
                }
            }
        }
    }

    fn parse_indexing(
        expr: Expression,
        tokens: &mut TokenStream,
    ) -> Result<Expression, ParseError> {
        assert_token!(tokens, Token::OpeningBracket);
        let num1 = parse_expression(tokens)?;
        let index = match_token!(tokens,
            Token::ClosingBracket => Index::Number(Box::new(num1)),
            Token::Colon => {
                let num2 = parse_expression(tokens)?;
                assert_token!(tokens, Token::ClosingBracket);
                Index::Range(Box::new(num1), Box::new(num2))
            },
        );
        Ok(Expression::Index(Box::new(expr), index))
    }

    fn parse_block(tokens: &mut TokenStream) -> Result<Expression, ParseError> {
        let rules = parse_statements(tokens)?;
        let expr = parse_expression(tokens)?;
        assert_token!(tokens, Token::ClosingBrace);
        Ok(Expression::Block(rules, Box::new(expr)))
    }

    fn parse_component(tokens: &mut TokenStream) -> Result<Expression, ParseError> {
        let mut args = vec![];
        loop {
            match_token!(tokens,
                Token::Ident(name) => args.push(name),
                Token::Bar => break,
            );
            match_token!(tokens,
                Token::Comma => (),
                Token::Bar => break,
            );
        }
        Ok(Expression::Component(
            args.into_iter()
                .map(|x| (x, Expression::VarAccess("any".into())))
                .map(Item::Value)
                .collect(),
            vec![],
            Box::new(Expression::VarAccess("any".into())),
            Box::new(parse_expression(tokens)?),
        ))
    }

    fn parse_statements(tokens: &mut TokenStream) -> Result<Vec<Statement>, ParseError> {
        iter::from_fn(|| try_parse_statement(tokens)).collect()
    }

    #[derive(Clone)]
    pub enum Structure {
        Tuple(Vec<Structure>),
        Value,
    }

    pub type ParseResult = (
        Vec<(Rc<str>, Structure)>,
        Vec<(Rc<str>, Structure)>,
        Vec<Statement>,
    );

    pub fn full_parse(string: &str) -> Result<ParseResult, ParseError> {
        let tokens = &mut TokenStream::new(string);
        assert_token!(tokens, Token::Keyword(Keyword::Input));
        let mut inputs = vec![];
        loop {
            match_token!(tokens,
                Token::Ident(name) => inputs.push((
                    name,
                    {
                        let mut curr = Structure::Value;
                        while let Ok(Token::OpeningBracket) = tokens.peek(){
                            assert_token!(tokens, Token::OpeningBracket);
                            curr = Structure::Tuple(vec![curr; match_token!(tokens, Token::Numeral(num) => num as usize)]);
                            assert_token!(tokens, Token::ClosingBracket);
                        }
                        curr
                    },
                )),
                Token::Semicolon => break,
            )
        }
        let (outputs, ret) = match tokens.peek() {
            Ok(Token::Keyword(Keyword::Output)) => {
                assert_token!(tokens, Token::Keyword(Keyword::Output));
                let mut outputs = vec![];
                loop {
                    match_token!(tokens,
                        Token::Ident(name) => outputs.push((
                            name,
                            {
                                let mut curr = Structure::Value;
                                while let Ok(Token::OpeningBracket) = tokens.peek(){
                                    assert_token!(tokens, Token::OpeningBracket);
                                    curr = Structure::Tuple(vec![curr; match_token!(tokens, Token::Numeral(num) => num as usize)]);
                                    assert_token!(tokens, Token::ClosingBracket);
                                }
                                curr
                            },
                        )),
                        Token::Semicolon => break,
                    )
                }
                let ret = parse_statements(tokens)?;
                (outputs, ret)
            }
            _ => {
                let ret = parse_statements(tokens)?;
                assert_token!(tokens, Token::Keyword(Keyword::Output));
                let outputs = iter::from_fn(|| {
                    (|| {
                        match_token!(tokens,
                            Token::Ident(name) => Ok(Some(name)),
                            Token::Semicolon => Ok(None)
                        )
                    })()
                    .transpose()
                })
                .map(|x| x.map(|x| (x, Structure::Value)))
                .collect::<Result<_, _>>()?;
                (outputs, ret)
            }
        };
        // println!("{:?}", inputs);
        // println!("{:?}", outputs);
        // println!("{:?}", ret);
        match tokens.token() {
            Err(TokenError::EoF) => Ok((inputs, outputs, ret)),
            Ok(token) => Err(ParseError::UnexpectedToken(
                token,
                vec!["Token::Identifier"],
                line!(),
            )),
            Err(err) => Err(ParseError::TokenError(err)),
        }
    }
}
mod compiler {
    use crate::parser::*;
    use std::collections::HashMap;
    use std::hash::Hash;
    use std::iter;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static LABELS: AtomicUsize = AtomicUsize::new(0);

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Primitive {
        Type,
        Bool,
        Numeral,
    }
    #[derive(Debug, Clone)]
    pub enum Type {
        Any,
        Generic(Rc<str>),
        Typedef(usize),
        Primitive(Primitive),
        Tuple(Vec<Type>),
        Function(Vec<Type>, Box<Type>),
    }
    impl From<Primitive> for Type {
        fn from(value: Primitive) -> Type {
            Type::Primitive(value)
        }
    }
    impl PartialEq<Type> for Type {
        fn eq(&self, other: &Type) -> bool {
            self.try_match(other).is_some()
        }
    }
    fn try_join(
        mut first: HashMap<Rc<str>, Type>,
        second: HashMap<Rc<str>, Type>,
    ) -> Option<HashMap<Rc<str>, Type>> {
        for (key, value) in second {
            if first.contains_key(&key) && first.get(&key) != Some(&value) {
                return None;
            } else {
                first.insert(key, value);
            }
        }
        Some(first)
    }
    impl Type {
        fn fill_in(self, generics: &HashMap<Rc<str>, Type>) -> Type {
            match self {
                Type::Generic(ref name) => generics.get(name).map(Clone::clone).unwrap_or(self),
                Type::Tuple(types) => {
                    Type::Tuple(types.into_iter().map(|x| x.fill_in(generics)).collect())
                }
                Type::Function(args, ret) => Type::Function(
                    args.into_iter().map(|x| x.fill_in(generics)).collect(),
                    Box::new(ret.fill_in(generics)),
                ),
                Type::Primitive(_) | Type::Any | Type::Typedef(_) => self,
            }
        }
        fn try_match(&self, other: &Type) -> Option<HashMap<Rc<str>, Type>> {
            if let Type::Any = self {
                return Some(HashMap::new());
            }
            if let Type::Any = other {
                return Some(HashMap::new());
            }
            if let Type::Generic(name) = self {
                return Some(iter::once((name.clone(), other.clone())).collect());
            }
            if let Type::Generic(name) = other {
                return Some(iter::once((name.clone(), self.clone())).collect());
            }
            match self {
                Type::Primitive(prim1) => match other {
                    Type::Primitive(prim2) => (prim1 == prim2).then(HashMap::new),
                    _ => None,
                },
                Type::Typedef(id1) => match other {
                    Type::Typedef(id2) => (id1 == id2).then(HashMap::new),
                    _ => None,
                },
                Type::Tuple(types1) => match other {
                    Type::Tuple(types2) => {
                        let mut hashmap = HashMap::new();
                        for (ty1, ty2) in types1.iter().zip(types2) {
                            let ty1 = ty1.clone().fill_in(&hashmap);
                            let ty2 = ty2.clone().fill_in(&hashmap);
                            hashmap = ty1.try_match(&ty2).and_then(|x| try_join(hashmap, x))?;
                        }
                        Some(hashmap)
                    }
                    _ => None,
                },
                Type::Function(args1, ret1) => match other {
                    Type::Function(args2, ret2) => {
                        let mut hashmap = HashMap::new();
                        for (ty1, ty2) in args1.iter().zip(args2) {
                            let ty1 = ty1.clone().fill_in(&hashmap);
                            let ty2 = ty2.clone().fill_in(&hashmap);
                            hashmap = ty1.try_match(&ty2).and_then(|x| try_join(hashmap, x))?;
                        }
                        let ret1 = ret1.clone().fill_in(&hashmap);
                        let ret2 = ret2.clone().fill_in(&hashmap);
                        ret1.try_match(&ret2).and_then(|x| try_join(hashmap, x))
                    }
                    _ => None,
                },
                Type::Generic(_) | Type::Any => unreachable!(),
            }
        }
    }
    #[derive(Debug, Clone)]
    pub enum TypeError {
        Custom(String),
        UndefinedIdentifier(Rc<str>),
        CannotIndex(Type),
        CannotCall(Type),
        MismatchedTypes(Type, Type),
        OutOfBoundsIndexing(usize, usize),
        MissingArgument(Type),
        ExtraArgument(Type),
    }

    impl std::fmt::Display for TypeError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self{
                TypeError::UndefinedIdentifier(name) => 
                    write!(formatter, "Undefined Identifier {name}\nNote: {name} may not be defined later in the program than it is used"),
                TypeError::CannotIndex(ty) => 
                    write!(formatter, "Cannot index thing of type {ty:?}"),
                TypeError::CannotCall(ty) => 
                    write!(formatter, "Cannot call thing of type {ty:?}"),
                TypeError::MismatchedTypes(type1, type2) => 
                    write!(formatter, "Expected {type1:?}, found {type2:?}"),
                TypeError::OutOfBoundsIndexing(len, index) => 
                    write!(formatter, "Index out of bounds: the len is {len} but the index is {index}"),
                TypeError::MissingArgument(arg) => write!(formatter, "Missing argument of type {arg:?}"),
                TypeError::ExtraArgument(arg) => write!(formatter, "Extra argument of type {arg:?}"),
                TypeError::Custom(message) => formatter.write_str(message),
            }
        }
    }
    #[derive(Debug, Clone, PartialEq)]
    struct PartiallyCompiledRule(pub Rc<str>, pub PartiallyCompiledExpression);
    #[derive(Debug, Clone, PartialEq)]
    enum PartiallyCompiledExpression {
        Input(usize),
        Boolean(bool),
        Number(u128),
        Type(Type),
        Label(usize, Box<PartiallyCompiledExpression>),
        Circular(Type, usize),
        Wrap(usize, Box<PartiallyCompiledExpression>),
        Select(
            Box<PartiallyCompiledExpression>,
            Box<PartiallyCompiledExpression>,
            Box<PartiallyCompiledExpression>,
        ),
        Tuple(Vec<PartiallyCompiledExpression>),
        Component(Vec<(Rc<str>, Type)>, Type, Box<Expression>),
    }

    impl FromIterator<PartiallyCompiledExpression> for PartiallyCompiledExpression {
        fn from_iter<T>(iter: T) -> PartiallyCompiledExpression
        where
            T: IntoIterator<Item = PartiallyCompiledExpression>,
        {
            PartiallyCompiledExpression::Tuple(iter.into_iter().collect())
        }
    }

    impl<I> FromIterator<I> for PartiallyCompiledExpression
    where
        I: IntoIterator<Item = PartiallyCompiledExpression>,
    {
        fn from_iter<T>(iter: T) -> PartiallyCompiledExpression
        where
            T: IntoIterator<Item = I>,
        {
            PartiallyCompiledExpression::Tuple(iter.into_iter().flatten().collect())
        }
    }

    impl PartiallyCompiledExpression {
        fn try_into_iter(
            self,
        ) -> Result<std::vec::IntoIter<PartiallyCompiledExpression>, TypeError> {
            match self {
                PartiallyCompiledExpression::Tuple(exprs) => Ok(exprs.into_iter()),
                _ => Err(TypeError::CannotIndex(find_type(&self))),
            }
        }
    }

    type Context = HashMap<Rc<str>, Vec<PartiallyCompiledExpression>>;

    fn add_to_context<U, T>(id: U, val: T, context: &mut HashMap<U, Vec<T>>)
    where
        U: Eq + Hash,
    {
        if let Some(ref mut parts) = context.get_mut(&id) {
            parts.push(val);
        } else {
            context.insert(id, vec![val]);
        }
    }

    fn find_type(expr: &PartiallyCompiledExpression) -> Type {
        use crate::compiler::PartiallyCompiledExpression as PCE;
        match expr {
            PCE::Boolean(_) => Type::Primitive(Primitive::Bool),
            PCE::Select(_, _, _) => Type::Primitive(Primitive::Bool),
            PCE::Input(_) => Type::Primitive(Primitive::Bool),
            PCE::Number(_) => Type::Primitive(Primitive::Numeral),
            PCE::Type(_) => Type::Primitive(Primitive::Type),
            PCE::Circular(ty, _) => ty.clone(),
            PCE::Wrap(id, _) => Type::Typedef(*id),
            PCE::Label(_, expr) => find_type(expr),
            PCE::Tuple(exprs) => Type::Tuple(exprs.iter().map(find_type).collect()),
            PCE::Component(args, ret, _) => Type::Function(
                args.iter().map(|x| x.1.clone()).collect(),
                Box::new(ret.clone()),
            ),
        }
    }

    fn evaluate_type(expr: PartiallyCompiledExpression) -> Result<Type, TypeError> {
        use crate::compiler::PartiallyCompiledExpression as PCE;
        Ok(match expr {
            PCE::Type(ty) => ty,
            PCE::Tuple(exprs) => Type::Tuple(
                exprs
                    .into_iter()
                    .map(evaluate_type)
                    .collect::<Result<_, _>>()?,
            ),
            PCE::Label(_, expr) => evaluate_type(*expr)?,
            expr => {
                return Err(TypeError::MismatchedTypes(
                    Type::Primitive(Primitive::Type),
                    find_type(&expr),
                ))
            }
        })
    }

    fn reduce_item(
        item: Item<Expression>,
        context: &Context,
        label: Option<usize>,
        typedefs: &mut Vec<Type>,
    ) -> Result<Vec<PartiallyCompiledExpression>, TypeError> {
        Ok(match item {
            Item::Value(expr) => vec![reduce_expr(expr, context, label, typedefs)?],
            Item::Spread(expr) => match reduce_expr(expr, context, label, typedefs)? {
                PartiallyCompiledExpression::Tuple(exprs) => exprs,
                expr => return Err(TypeError::CannotIndex(find_type(&expr))),
            },
        })
    }

    fn reduce_expr(
        expr: Expression,
        context: &Context,
        label: Option<usize>,
        typedefs: &mut Vec<Type>,
    ) -> Result<PartiallyCompiledExpression, TypeError> {
        use crate::compiler::PartiallyCompiledExpression as PCE;
        use crate::parser::Expression as E;
        Ok(match expr {
            E::IsZero(expr) => match reduce_expr(*expr, context, label, typedefs)? {
                PCE::Number(num) => PCE::Boolean(num == 0),
                _ => todo!(),
            },
            E::Inc(expr) => match reduce_expr(*expr, context, label, typedefs)? {
                PCE::Number(num) => PCE::Number(num + 1),
                _ => todo!(),
            },
            E::Dec(expr) => match reduce_expr(*expr, context, label, typedefs)? {
                PCE::Number(0) => return Err(TypeError::Custom("Cannot decrement 0".into())),
                PCE::Number(num) => PCE::Number(num - 1),
                _ => todo!(),
            },
            E::Array(expr, reps) => match reduce_expr(*reps, context, label, typedefs)? {
                PCE::Number(reps) => PCE::Tuple(vec![
                    reduce_expr(*expr, context, label, typedefs)?;
                    reps as usize
                ]),
                _ => todo!(),
            },
            E::Map(name, expr, body) => reduce_expr(*expr, context, label, typedefs)?
                .try_into_iter()?
                .map(|x| {
                    let mut context = context.clone();
                    add_to_context(name.clone(), x, &mut context);
                    reduce_expr((*body).clone(), &context, label, typedefs)
                })
                .collect::<Result<_, _>>()?,
            E::Wrap(id, expr) => {
                PCE::Wrap(id, Box::new(reduce_expr(*expr, context, label, typedefs)?))
            }
            E::Unwrap(expr) => match reduce_expr(*expr, context, label, typedefs)? {
                PCE::Wrap(_, expr) => *expr,
                expr => {
                    return Err(TypeError::Custom(format!(
                        "Error: Can only unwrap a newtype, not a {:?}",
                        find_type(&expr)
                    )))
                }
            },
            E::Number(val, size) => match size {
                Some(size) => (0..size)
                    .map(|x| PCE::Boolean((val >> x) & 1 != 0))
                    .collect(),
                None => PCE::Number(val),
            },
            E::VarAccess(name) => context
                .get(&name)
                .and_then(|x| x.last())
                .ok_or(TypeError::UndefinedIdentifier(name))?
                .clone(),
            E::TypedVarAccess(name, ty) => {
                let ty = reduce_expr(*ty, context, None, typedefs).and_then(evaluate_type)?;
                let exprs = context
                    .get(&name)
                    .ok_or(TypeError::UndefinedIdentifier(name.clone()))?;
                let potential_ty = exprs
                    .last()
                    .map(find_type)
                    .ok_or(TypeError::UndefinedIdentifier(name))?;
                exprs
                    .iter()
                    .filter(|x| find_type(x) == ty)
                    .last()
                    .ok_or(TypeError::MismatchedTypes(ty, potential_ty))?
                    .clone()
            }
            E::Select(a, b, c) => match reduce_expr(*a, context, label, typedefs)? {
                PCE::Boolean(val) => {
                    if val {
                        reduce_expr(*b, context, label, typedefs)?
                    } else {
                        reduce_expr(*c, context, label, typedefs)?
                    }
                }
                a => {
                    let b = reduce_expr(*b, context, label, typedefs)?;
                    let c = reduce_expr(*c, context, label, typedefs)?;
                    if b == c {
                        b
                    } else {
                        PCE::Select(Box::new(a), Box::new(b), Box::new(c))
                    }
                }
            },
            E::Typedef(ty) => {
                let ty = reduce_expr(*ty, context, None, typedefs).and_then(evaluate_type)?;
                typedefs.push(ty);
                PCE::Type(Type::Typedef(typedefs.len() - 1))
            }
            E::CompType(args, ret) => PCE::Type(Type::Function(
                args.into_iter()
                    .map(|x| reduce_expr(x, context, None, typedefs).and_then(evaluate_type))
                    .collect::<Result<_, _>>()?,
                reduce_expr(*ret, context, None, typedefs)
                    .and_then(evaluate_type)
                    .map(Box::new)?,
            )),
            E::Tuple(exprs) => exprs
                .into_iter()
                .map(|x| reduce_item(x, context, label, typedefs))
                .collect::<Result<_, _>>()?,
            E::Index(expr, index) => match reduce_expr(*expr, context, label, typedefs)? {
                PCE::Tuple(mut exprs) => match index {
                    Index::Number(index) => match reduce_expr(*index, context, label, typedefs)? {
                        PCE::Number(index) => {
                            if index as usize >= exprs.len() {
                                return Err(TypeError::OutOfBoundsIndexing(
                                    exprs.len(),
                                    index as usize,
                                ));
                            }
                            exprs.swap_remove(index as usize)
                        }
                        _ => todo!(),
                    },
                    Index::Range(_start, _end) => todo!(),
                },
                // expr => todo!("{expr:?}"),
                expr => return Err(TypeError::CannotIndex(find_type(&expr))),
            },
            E::Block(rules, expr) => reduce_block(rules, *expr, context, typedefs)?,
            E::Call(func, args) => reduce_call(*func, args, context, label, typedefs)?,
            E::Component(args, generics, ret, expr) => {
                let mut context = context.clone();
                context.extend(
                    generics
                        .into_iter()
                        .map(|x| (x.clone(), vec![PCE::Type(Type::Generic(x))])),
                );
                PCE::Component(
                    args.into_iter()
                        .map(Item::unwrap)
                        .map(|(n, x)| {
                            reduce_expr(x, &context, None, typedefs)
                                .and_then(evaluate_type)
                                .map(|x| (n, x))
                        })
                        .collect::<Result<_, _>>()?,
                    reduce_expr(*ret, &context, None, typedefs).and_then(evaluate_type)?,
                    expr,
                )
            }
            E::Error(message) => {
                return Err(TypeError::Custom(format!("Error with \"{message}\"")))
            }
            E::String(string) => string
                .as_bytes()
                .iter()
                .map(|x| (0..8).map(move |i| PCE::Boolean((*x >> i) & 1 != 0)))
                .collect(),
            E::Circular(ty) => PCE::Circular(
                reduce_expr(*ty, context, None, typedefs).and_then(evaluate_type)?,
                label.ok_or(TypeError::Custom(
                    "Cannot use circular expression outside of a rule or in a type".into(),
                ))?,
            ),
        })
    }

    fn reduce_call(
        func: Expression,
        args: Vec<Item<Expression>>,
        context: &Context,
        label: Option<usize>,
        typedefs: &mut Vec<Type>,
    ) -> Result<PartiallyCompiledExpression, TypeError> {
        let ((arg_names, mut arg_types), ret, expr): ((Vec<_>, Vec<_>), _, _) =
            match reduce_expr(func, context, label, typedefs)? {
                PartiallyCompiledExpression::Component(a, b, c) => (a.into_iter().unzip(), b, *c),
                PartiallyCompiledExpression::Type(Type::Typedef(id)) => (
                    (vec!["*a".into()], vec![typedefs[id].clone()]),
                    Type::Typedef(id),
                    Expression::Wrap(id, Box::new(Expression::VarAccess("*a".into()))),
                ),
                expr => return Err(TypeError::CannotCall(find_type(&expr))),
            };
        if arg_types.len() > args.len() {
            return Err(TypeError::MissingArgument(
                arg_types.swap_remove(args.len()),
            ));
        }
        let args = {
            let mut result = vec![];
            args.into_iter().try_for_each(|x| {
                reduce_item(x, context, label, typedefs).map(|mut x| result.append(&mut x))
            })?;
            result
        };
        if args.len() > arg_types.len() {
            return Err(TypeError::ExtraArgument(find_type(&args[arg_types.len()])));
        }
        let mut resolved = HashMap::new();
        for (t1, t2) in arg_types.into_iter().zip(args.iter().map(find_type)) {
            let t1 = t1.fill_in(&resolved);
            let t2 = t2.fill_in(&resolved);
            resolved = t1
                .try_match(&t2)
                .and_then(|x| try_join(resolved, x))
                .ok_or(TypeError::MismatchedTypes(t1, t2))?;
        }
        let ret = ret.fill_in(&resolved);
        let mut context = context.clone();
        arg_names
            .into_iter()
            .zip(args)
            .for_each(|(n, v)| add_to_context(n, v, &mut context));
        resolved.into_iter().for_each(|(n, t)| {
            add_to_context(n, PartiallyCompiledExpression::Type(t), &mut context)
        });
        let expr = reduce_expr(expr, &context, None, typedefs)?;
        let ret_actual = find_type(&expr);
        if ret != ret_actual {
            return Err(TypeError::MismatchedTypes(ret, ret_actual));
        };
        Ok(expr)
    }

    fn reduce_block(
        rules: Vec<Statement>,
        expr: Expression,
        context: &Context,
        typedefs: &mut Vec<Type>,
    ) -> Result<PartiallyCompiledExpression, TypeError> {
        let mut context = context.clone();
        reduce_rules(rules, &mut context, typedefs)?;
        reduce_expr(expr, &context, None, typedefs)
    }

    fn reduce_rule(
        rule: Rule,
        context: &mut Context,
        typedefs: &mut Vec<Type>,
    ) -> Result<(), TypeError> {
        let expr = if rule.2 {
            let label = LABELS.fetch_add(1, Ordering::Relaxed);
            PartiallyCompiledExpression::Label(
                label,
                Box::new(reduce_expr(rule.3, &*context, Some(label), typedefs)?),
            )
        } else {
            reduce_expr(rule.3, &*context, None, typedefs)?
        };
        let expr_type = find_type(&expr);
        let mut reqs = rule
            .1
            .into_iter()
            .map(|x| reduce_expr(x, &*context, None, typedefs).and_then(evaluate_type));
        if let Some(t) = reqs.find(|x| x.as_ref().map_or(true, |x| x != &expr_type)) {
            t.and_then(|x| Err(TypeError::MismatchedTypes(x, expr_type)))?;
        }
        add_to_context(rule.0, expr, context);
        Ok(())
    }

    fn reduce_rules(
        rules: Vec<Statement>,
        context: &mut Context,
        typedefs: &mut Vec<Type>,
    ) -> Result<(), TypeError> {
        rules.into_iter().try_for_each(|x| match x {
            Statement::Rule(rule) => reduce_rule(rule, context, typedefs),
            Statement::Foreach(Foreach(name, expr, rules)) => {
                reduce_expr(expr, &*context, None, typedefs)?
                    .try_into_iter()?
                    .try_for_each(|x| {
                        add_to_context(name.clone(), x, context);
                        reduce_rules(rules.clone(), context, typedefs)
                    })
            }
            Statement::If(If(cond, rules)) => match reduce_expr(cond, &*context, None, typedefs)? {
                PartiallyCompiledExpression::Boolean(true) => {
                    reduce_rules(rules, context, typedefs)
                }
                PartiallyCompiledExpression::Boolean(false) => Ok(()),
                _ => todo!(),
            },
        })
    }

    fn structure_to_pce(structure: Structure, id: &mut usize) -> PartiallyCompiledExpression {
        match structure {
            Structure::Value => {
                let expr = PartiallyCompiledExpression::Input(*id);
                *id += 1;
                expr
            }
            Structure::Tuple(structures) => structures
                .into_iter()
                .map(|x| structure_to_pce(x, id))
                .collect(),
        }
    }

    fn reduce(
        rules: Vec<Statement>,
        inputs: Vec<(Rc<str>, Structure)>,
    ) -> Result<Context, TypeError> {
        use crate::compiler::PartiallyCompiledExpression as PCE;
        let mut context = HashMap::with_capacity(inputs.len() + 5);
        inputs
            .into_iter()
            .for_each(|(n, s)| add_to_context(n, structure_to_pce(s, &mut 0), &mut context));
        add_to_context("false".into(), PCE::Boolean(false), &mut context);
        add_to_context("true".into(), PCE::Boolean(true), &mut context);
        add_to_context(
            "bool".into(),
            PCE::Type(Primitive::Bool.into()),
            &mut context,
        );
        add_to_context(
            "type".into(),
            PCE::Type(Primitive::Type.into()),
            &mut context,
        );
        add_to_context(
            "numeral".into(),
            PCE::Type(Primitive::Numeral.into()),
            &mut context,
        );
        add_to_context("any".into(), PCE::Type(Type::Any), &mut context);
        add_to_context(
            "select".into(),
            PCE::Component(
                vec![
                    ("a".into(), Type::Primitive(Primitive::Bool)),
                    ("b".into(), Type::Primitive(Primitive::Bool)),
                    ("c".into(), Type::Primitive(Primitive::Bool)),
                ],
                Type::Primitive(Primitive::Bool),
                Box::new(Expression::Select(
                    Box::new(Expression::VarAccess("a".into())),
                    Box::new(Expression::VarAccess("b".into())),
                    Box::new(Expression::VarAccess("c".into())),
                )),
            ),
            &mut context,
        );
        add_to_context(
            "inc".into(),
            PCE::Component(
                vec![("a".into(), Type::Primitive(Primitive::Numeral))],
                Type::Primitive(Primitive::Numeral),
                Box::new(Expression::Inc(Box::new(Expression::VarAccess("a".into())))),
            ),
            &mut context,
        );
        add_to_context(
            "dec".into(),
            PCE::Component(
                vec![("a".into(), Type::Primitive(Primitive::Numeral))],
                Type::Primitive(Primitive::Numeral),
                Box::new(Expression::Dec(Box::new(Expression::VarAccess("a".into())))),
            ),
            &mut context,
        );
        add_to_context(
            "is_zero".into(),
            PCE::Component(
                vec![("a".into(), Type::Primitive(Primitive::Numeral))],
                Type::Primitive(Primitive::Bool),
                Box::new(Expression::IsZero(Box::new(Expression::VarAccess(
                    "a".into(),
                )))),
            ),
            &mut context,
        );
        reduce_rules(rules, &mut context, &mut vec![])?;
        Ok(context)
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum Value {
        Boolean(bool),
        Input(usize),
        Circular(usize),
        Label(usize, Box<Value>),
        Select(Box<Value>, Box<Value>, Box<Value>),
    }

    fn resolve_expr(expr: PartiallyCompiledExpression) -> Result<Value, TypeError> {
        use crate::compiler::PartiallyCompiledExpression as PCE;
        use crate::compiler::Value as V;
        Ok(match expr {
            PCE::Boolean(val) => V::Boolean(val),
            PCE::Input(id) => V::Input(id),
            PCE::Circular(Type::Primitive(Primitive::Bool), id) => V::Circular(id),
            PCE::Label(id, expr) => V::Label(id, Box::new(resolve_expr(*expr)?)),
            PCE::Select(a, b, c) => V::Select(
                Box::new(resolve_expr(*a)?),
                Box::new(resolve_expr(*b)?),
                Box::new(resolve_expr(*c)?),
            ),
            expr => todo!("{expr:#?}"),
        })
    }

    fn resolve_expr_structural(
        expr: PartiallyCompiledExpression,
        structure: Structure,
    ) -> Result<Vec<Value>, TypeError> {
        match structure {
            Structure::Value => resolve_expr(expr).map(|x| vec![x]),
            Structure::Tuple(structures) => match expr {
                PartiallyCompiledExpression::Tuple(exprs) => {
                    let mut result = vec![];
                    for (expr, structure) in exprs.into_iter().zip(structures) {
                        result.append(&mut resolve_expr_structural(expr, structure)?)
                    }
                    Ok(result)
                }
                expr => todo!("{expr:#?}"),
            },
        }
    }

    fn resolve(
        mut context: Context,
        outputs: Vec<(Rc<str>, Structure)>,
    ) -> Result<Vec<Value>, TypeError> {
        let mut result = vec![];
        for (x, s) in outputs {
            let exprs = context
                .remove(&x)
                .ok_or(TypeError::UndefinedIdentifier(x.clone()))?;
            let potential_ty = exprs
                .last()
                .map(find_type)
                .ok_or(TypeError::UndefinedIdentifier(x))?;
            let t = find_type(&structure_to_pce(s.clone(), &mut 0));
            match exprs.into_iter().filter(|x| find_type(x) == t).last() {
                Some(expr) => result.append(&mut resolve_expr_structural(expr, s)?),
                None => return Err(TypeError::MismatchedTypes(t, potential_ty)),
            };
        }
        Ok(result)
    }

    pub fn compile((inputs, outputs, statements): ParseResult) -> Result<Vec<Value>, TypeError> {
        resolve(reduce(statements, inputs)?, outputs)
    }
}

fn main() {
    let string = "
INPUT;
OUTPUT cells;
N = 10;
comp Range(n: numeral) -> any {
    i = 0;
    l = [];
    foreach _: [bool; n] {
        l = [..l, i];
        i = inc(i);
    }
    l
}
comp or(vals: any) -> bool {
    so_far = false;
    foreach val: vals {
        so_far = select(val, true, so_far);
    }
    so_far
}
comp and(vals: any) -> bool {
    so_far = true;
    foreach val: vals {
        so_far = select(val, so_far, false);
    }
    so_far
}
comp not(val: bool) -> bool {
    select(val, false, true)
}
comp cell(
    up_left: bool,
    up: bool,
    up_right: bool,
    left_left: bool,
    left: bool,
    center: bool,
    right: bool,
    down_left: bool,
    down: bool,
    down_right: bool) -> bool {
    val = or(
        up,
        and(up_left, left, left_left),
        and(up_right, right),
        and(center, not(or(down_left, down, down_right))));
    val
}
comp eq(n1: numeral, n2: numeral) -> bool {
    if is_zero(n1) {
        val = is_zero(n2);
    }
    if is_zero(n2) {
        val = is_zero(n1);
    }
    if and([not(is_zero(n1)), not(is_zero(n2))]) {
        val = eq(dec(n1), dec(n2));
    }
    val
}
comp update_cells(cells: [[bool; N]; N]) -> [[bool; N]; N] {
    map x: Range(10) map y: Range(10) {
        up_left = false;
        up = false;
        up_right = false;
        left_left = false;
        left = false;
        center = false;
        right = false;
        down_left = false;
        down = false;
        down_right = false;
        if not(is_zero(x)) {
            left = cells[Pred(x)][y];
            if Gt(y, 0) {
                up_left = cells[Pred(x)][Pred(y)];
            }
            if Lt(y, Pred(N)) {
                down_left = cells[Pred(x)][Succ(y)];
            }
            if Gt(x, 1) {
                left_left = cells[Pred(Pred(x))][y];
            }
        }
        if not(eq(inc(x), N)) {
            right = cells[Succ(x)][y];
            if Gt(y, 0) {
                up_right = cells[Succ(x)][Pred(y)];
            }
            if Lt(y, Pred(H)) {
                down_right = cells[Succ(x)][Succ(y)];
            }
        }
        center = cells[x][y];
        if Gt(y, 0) {
            up = cells[x][Pred(y)];
        }
        if Lt(y, Pred(H)) {
            down = cells[x][Succ(y)];
        }
        cell(
            up_left,
            up,
            up_right,
            left_left,
            left,
            center,
            right,
            down_left,
            down,
            down_right,
        )
    }
}
cells = update_cells(@[[bool; N]; N]);
    ";
    match full_parse(string).map(compile) {
        Ok(Ok(rules)) => println!("{rules:#?}"),
        Ok(Err(err)) => println!("{err}"),
        Err(err) => println!("{err}"),
    }
}

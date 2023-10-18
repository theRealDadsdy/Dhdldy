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
            "typedef" => Some(Keyword::TypeDef),
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
        Bar,
        Comma,
        Semicolon,
        Colon,
        Assoc,
        Arrow,
        Equals,
        At,
        Dollar,
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
                Token::Bar => write!(formatter, "|"),
                Token::Comma => write!(formatter, ","),
                Token::Colon => write!(formatter, ":"),
                Token::Assoc => write!(formatter, "::"),
                Token::Semicolon => write!(formatter, ";"),
                Token::Arrow => write!(formatter, "->"),
                Token::Equals => write!(formatter, "="),
                Token::At => write!(formatter, "@"),
                Token::Dollar => write!(formatter, "$"),
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
                    '-' => match self.0.next() {
                        Some('>') => Token::Arrow,
                        Some(_) => Err(TokenError::InvalidChar('-'))?,
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
    use crate::tokenizer::Keyword;
    use crate::tokenizer::Token;
    use crate::tokenizer::TokenError;
    use crate::tokenizer::TokenStream;
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
    pub struct Rule(pub Rc<str>, pub Vec<Expression>, pub Expression);
    #[derive(Debug, Clone, PartialEq)]
    pub enum Expression {
        String(String),
        Error(String),
        Select(Box<Expression>, Box<Expression>, Box<Expression>),
        VarAccess(Rc<str>),
        TypedVarAccess(Rc<str>, Box<Expression>),
        Circular(Box<Expression>),
        Tuple(Vec<Expression>),
        Index(Box<Expression>, Index),
        Call(Box<Expression>, Vec<Expression>),
        Block(Vec<Rule>, Box<Expression>),
        Component(
            Vec<(Rc<str>, Expression)>,
            Vec<Rc<str>>,
            Box<Expression>,
            Box<Expression>,
        ),
        CompType(Vec<Expression>, Box<Expression>),
        Number(u128, Option<u32>),
    }
    #[derive(Debug, Clone, PartialEq)]
    pub enum Index {
        Number(usize),
        Range(usize, usize),
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

    fn try_parse_statement(tokens: &mut TokenStream) -> Option<Result<Rule, ParseError>> {
        match (tokens.peek(), tokens.peek2()) {
            (Ok(Token::Keyword(Keyword::Error)), _) => Some({
                (|| {
                    let error = parse_expression(tokens)?;
                    assert_token!(tokens, Token::Semicolon);
                    Ok(Rule("*".into(), vec![], error))
                })()
            }),
            (Ok(Token::Ident(_)), Ok(Token::Colon | Token::Equals)) => Some(parse_rule(tokens)),
            (Ok(Token::Keyword(Keyword::FuncDef)), _) => Some(parse_func_def(tokens)),
            _ => None,
        }
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
                    args.push((name, parse_expression(tokens)?));
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
        while let Some(expr) = subs.pop() {
            use Expression as E;
            match expr {
                E::Tuple(mut exprs) => subs.append(&mut exprs),
                E::Call(expr, mut exprs) => {
                    subs.push(*expr);
                    subs.append(&mut exprs);
                }
                E::Circular(ty) => reqs.push(*ty),
                E::Index(expr, _) => subs.push(*expr),
                E::Number(..)
                | E::String(_)
                | E::VarAccess(_)
                | E::TypedVarAccess(_, _)
                | E::Component(..)
                | E::Block(..) => (),
                expr => todo!("{expr:?}"),
            }
        }
        Ok(Rule(name, reqs, expr))
    }

    fn parse_expression(tokens: &mut TokenStream) -> Result<Expression, ParseError> {
        let mut res = match_token!(tokens,
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
                _ => return Ok(res),
            }
        }
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
                    exprs.push(parse_expression(tokens)?);
                    match_token!(tokens,
                        Token::Comma => {},
                        Token::Semicolon => {
                            let reps = match_token!(tokens, Token::Numeral(num) => num);
                            let expr = exprs.pop().expect("I just added it, grr");
                            assert_token!(tokens, Token::ClosingBracket);
                            return Ok(Expression::Tuple(vec![expr; reps as usize]));
                        },
                        Token::ClosingBracket => return Ok(Expression::Tuple(exprs)),
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
                    Token::Comma => exprs.push(expr),
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
                    exprs.push(parse_expression(tokens)?);
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
                    exprs.push(parse_expression(tokens)?);
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
        let num1 = match_token!(tokens,
            Token::Numeral(num) => num.try_into().map_err(|_| ParseError::InvalidNumeral(num))?,
        );
        let index = match_token!(tokens,
            Token::ClosingBracket => Index::Number(num1),
            Token::Colon => {
                let num2 = match_token!(tokens,
                    Token::Numeral(num) => num.try_into().map_err(|_| ParseError::InvalidNumeral(num))?,
                );
                assert_token!(tokens, Token::ClosingBracket);
                Index::Range(num1, num2)
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
                .collect(),
            vec![],
            Box::new(Expression::VarAccess("any".into())),
            Box::new(parse_expression(tokens)?),
        ))
    }

    fn parse_statements(tokens: &mut TokenStream) -> Result<Vec<Rule>, ParseError> {
        std::iter::from_fn(|| try_parse_statement(tokens)).collect()
    }

    pub type ParseResult = (Vec<Rc<str>>, Vec<Rc<str>>, Vec<Rule>);

    pub fn full_parse(string: &str) -> Result<ParseResult, ParseError> {
        let tokens = &mut TokenStream::new(string);
        assert_token!(tokens, Token::Keyword(Keyword::Input));
        let inputs = std::iter::from_fn(|| {
            (|| {
                match_token!(tokens,
                    Token::Ident(name) => Ok(Some(name)),
                    Token::Semicolon => Ok(None)
                )
            })()
            .transpose()
        })
        .collect::<Result<_, _>>()?;
        let (outputs, ret) = match tokens.peek() {
            Ok(Token::Keyword(Keyword::Output)) => {
                assert_token!(tokens, Token::Keyword(Keyword::Output));
                let outputs = std::iter::from_fn(|| {
                    (|| {
                        match_token!(tokens,
                            Token::Ident(name) => Ok(Some(name)),
                            Token::Semicolon => Ok(None)
                        )
                    })()
                    .transpose()
                })
                .collect::<Result<_, _>>()?;
                let ret = parse_statements(tokens)?;
                (outputs, ret)
            }
            _ => {
                let ret = parse_statements(tokens)?;
                assert_token!(tokens, Token::Keyword(Keyword::Output));
                let outputs = std::iter::from_fn(|| {
                    (|| {
                        match_token!(tokens,
                            Token::Ident(name) => Ok(Some(name)),
                            Token::Semicolon => Ok(None)
                        )
                    })()
                    .transpose()
                })
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
    use core::hash::Hash;
    use std::collections::HashMap;
    use std::rc::Rc;

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
                Type::Primitive(_) | Type::Any => self,
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
                return Some(std::iter::once((name.clone(), other.clone())).collect());
            }
            if let Type::Generic(name) = other {
                return Some(std::iter::once((name.clone(), self.clone())).collect());
            }
            match self {
                Type::Primitive(prim1) => match other {
                    Type::Primitive(prim2) => (prim1 == prim2).then(HashMap::new),
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
                Type::Function(_args, _ret) => todo!(),
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
                TypeError::Custom(message) => write!(formatter, "Error \"{message}\""),
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
        Select(
            Box<PartiallyCompiledExpression>,
            Box<PartiallyCompiledExpression>,
            Box<PartiallyCompiledExpression>,
        ),
        // Circular(Type),
        Tuple(Vec<PartiallyCompiledExpression>),
        Component(Vec<(Rc<str>, Type)>, Type, Box<Expression>),
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
        use PartiallyCompiledExpression as PCE;
        match expr {
            PCE::Boolean(_) => Type::Primitive(Primitive::Bool),
            PCE::Input(_) => Type::Primitive(Primitive::Bool),
            PCE::Select(_, _, _) => Type::Primitive(Primitive::Bool),
            PCE::Number(_) => Type::Primitive(Primitive::Numeral),
            PCE::Type(_) => Type::Primitive(Primitive::Type),
            PCE::Tuple(exprs) => Type::Tuple(exprs.iter().map(find_type).collect()),
            PCE::Component(args, ret, _) => Type::Function(
                args.iter().map(|x| x.1.clone()).collect(),
                Box::new(ret.clone()),
            ),
        }
    }

    fn evaluate_type(expr: PartiallyCompiledExpression) -> Result<Type, TypeError> {
        use PartiallyCompiledExpression as PCE;
        Ok(match expr {
            PCE::Type(ty) => ty,
            PCE::Tuple(exprs) => Type::Tuple(
                exprs
                    .into_iter()
                    .map(evaluate_type)
                    .collect::<Result<_, _>>()?,
            ),
            expr => {
                return Err(TypeError::MismatchedTypes(
                    Type::Primitive(Primitive::Type),
                    find_type(&expr),
                ))
            }
        })
    }

    fn reduce_expr(
        expr: Expression,
        context: &Context,
    ) -> Result<PartiallyCompiledExpression, TypeError> {
        use Expression as E;
        use PartiallyCompiledExpression as PCE;
        Ok(match expr {
            E::Number(val, size) => match size {
                Some(size) => PCE::Tuple(
                    (0..size)
                        .map(|x| (val >> x) != 0)
                        .map(PCE::Boolean)
                        .collect(),
                ),
                None => PCE::Number(val),
            },
            E::VarAccess(name) => context
                .get(&name)
                .and_then(|x| x.last())
                .ok_or(TypeError::UndefinedIdentifier(name))?
                .clone(),
            E::TypedVarAccess(name, ty) => {
                let ty = reduce_expr(*ty, context).and_then(evaluate_type)?;
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
            E::Select(a, b, c) => match reduce_expr(*a, context)? {
                PCE::Boolean(val) => {
                    if val {
                        reduce_expr(*b, context)?
                    } else {
                        reduce_expr(*c, context)?
                    }
                }
                a => {
                    let b = reduce_expr(*b, context)?;
                    let c = reduce_expr(*c, context)?;
                    if b == c {
                        b
                    } else {
                        PCE::Select(Box::new(a), Box::new(b), Box::new(c))
                    }
                }
            },
            E::CompType(args, ret) => PCE::Type(Type::Function(
                args.into_iter()
                    .map(|x| reduce_expr(x, context).and_then(evaluate_type))
                    .collect::<Result<_, _>>()?,
                reduce_expr(*ret, context)
                    .and_then(evaluate_type)
                    .map(Box::new)?,
            )),
            E::Tuple(exprs) => PCE::Tuple(
                exprs
                    .into_iter()
                    .map(|x| reduce_expr(x, context))
                    .collect::<Result<_, _>>()?,
            ),
            E::Index(expr, index) => match reduce_expr(*expr, context)? {
                PCE::Tuple(mut exprs) => match index {
                    Index::Number(index) => {
                        if index >= exprs.len() {
                            return Err(TypeError::OutOfBoundsIndexing(exprs.len(), index));
                        }
                        exprs.swap_remove(index)
                    }
                    Index::Range(_start, _end) => todo!(),
                },
                // expr => todo!("{expr:?}"),
                expr => return Err(TypeError::CannotIndex(find_type(&expr))),
            },
            E::Block(rules, expr) => reduce_block(rules, *expr, context)?,
            E::Call(func, args) => reduce_call(*func, args, context)?,
            E::Component(args, generics, ret, expr) => {
                let mut context = context.clone();
                context.extend(
                    generics
                        .into_iter()
                        .map(|x| (x.clone(), vec![PCE::Type(Type::Generic(x))])),
                );
                PCE::Component(
                    args.into_iter()
                        .map(|(n, x)| {
                            reduce_expr(x, &context)
                                .and_then(evaluate_type)
                                .map(|x| (n, x))
                        })
                        .collect::<Result<_, _>>()?,
                    reduce_expr(*ret, &context).and_then(evaluate_type)?,
                    expr,
                )
            }
            E::Error(message) => return Err(TypeError::Custom(message)),
            E::String(string) => PCE::Tuple(
                string
                    .chars()
                    .flat_map(|x| (0..8).map(move |i| (x as u8) >> i != 0))
                    .map(PCE::Boolean)
                    .collect(),
            ),
            expr => todo!("{expr:#?}\n{context:?}"),
        })
    }

    fn reduce_call(
        func: Expression,
        args: Vec<Expression>,
        context: &Context,
    ) -> Result<PartiallyCompiledExpression, TypeError> {
        let ((arg_names, mut arg_types), ret, expr): ((Vec<_>, Vec<_>), _, _) =
            match reduce_expr(func, context)? {
                PartiallyCompiledExpression::Component(a, b, c) => (a.into_iter().unzip(), b, *c),
                expr => return Err(TypeError::CannotCall(find_type(&expr))),
            };
        if arg_types.len() > args.len() {
            return Err(TypeError::MissingArgument(
                arg_types.swap_remove(args.len()),
            ));
        }
        let args = args
            .into_iter()
            .map(|x| reduce_expr(x, context))
            .collect::<Result<Vec<_>, _>>()?;
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
        let expr = reduce_expr(expr, &context)?;
        let ret_actual = find_type(&expr);
        if ret != ret_actual {
            return Err(TypeError::MismatchedTypes(ret, ret_actual));
        };
        Ok(expr)
    }

    fn reduce_block(
        rules: Vec<Rule>,
        expr: Expression,
        context: &Context,
    ) -> Result<PartiallyCompiledExpression, TypeError> {
        let mut context = context.clone();
        reduce_rules(rules, &mut context)?;
        reduce_expr(expr, &context)
    }

    fn reduce_rules(rules: Vec<Rule>, context: &mut Context) -> Result<(), TypeError> {
        rules.into_iter().try_for_each(|x| {
            let mut reqs =
                x.1.into_iter()
                    .map(|x| reduce_expr(x, &*context).and_then(evaluate_type));
            let expr = reduce_expr(x.2, &*context)?;
            let expr_type = find_type(&expr);
            if let Some(t) = reqs.find(|x| x.as_ref().map_or(true, |x| x != &expr_type)) {
                t.and_then(|x| Err(TypeError::MismatchedTypes(x, expr_type)))?;
            }
            add_to_context(x.0, expr, context);
            Ok(())
        })
    }

    fn reduce(rules: Vec<Rule>, inputs: Vec<Rc<str>>) -> Result<Context, TypeError> {
        use PartiallyCompiledExpression as PCE;
        let mut context = HashMap::with_capacity(inputs.len() + 5);
        inputs
            .into_iter()
            .enumerate()
            .for_each(|(i, x)| add_to_context(x, PCE::Input(i), &mut context));
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
        reduce_rules(rules, &mut context)?;
        Ok(context)
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum Value {
        Boolean(bool),
        Input(usize),
        Select(Box<Value>, Box<Value>, Box<Value>),
    }

    fn resolve_expr(expr: PartiallyCompiledExpression) -> Result<Value, TypeError> {
        use PartiallyCompiledExpression as PCE;
        use Value as V;
        Ok(match expr {
            PCE::Boolean(val) => V::Boolean(val),
            PCE::Input(val) => V::Input(val),
            PCE::Select(a, b, c) => match resolve_expr(*a)? {
                V::Boolean(val) => {
                    if val {
                        resolve_expr(*b)?
                    } else {
                        resolve_expr(*c)?
                    }
                }
                a => {
                    let b = resolve_expr(*b)?;
                    let c = resolve_expr(*c)?;
                    if b == c {
                        b
                    } else {
                        V::Select(Box::new(a), Box::new(b), Box::new(c))
                    }
                }
            },
            expr @ PCE::Number(_) => {
                return Err(TypeError::MismatchedTypes(
                    Type::Primitive(Primitive::Bool),
                    find_type(&expr),
                ))
            }
            expr => todo!("{expr:#?}"),
        })
    }

    fn resolve(
        mut context: Context,
        outputs: Vec<(Rc<str>, Type)>,
    ) -> Result<Vec<Value>, TypeError> {
        outputs
            .into_iter()
            .map(|(x, t)| {
                let exprs = context
                    .remove(&x)
                    .ok_or(TypeError::UndefinedIdentifier(x.clone()))?;
                let potential_ty = exprs
                    .last()
                    .map(find_type)
                    .ok_or(TypeError::UndefinedIdentifier(x))?;
                match exprs.into_iter().filter(|x| find_type(x) == t).last() {
                    Some(expr) => resolve_expr(expr),
                    None => Err(TypeError::MismatchedTypes(t, potential_ty)),
                }
            })
            .collect()
    }

    pub fn compile((inputs, outputs, statements): ParseResult) -> Result<Vec<Value>, TypeError> {
        resolve(
            reduce(statements, inputs)?,
            outputs
                .into_iter()
                .zip(std::iter::repeat(Type::Primitive(Primitive::Bool)))
                .collect(),
        )
    }
}

fn main() {
    let string = "
    INPUT;
    OUTPUT o;
    comp typeof<N>(val: N) -> type {
        N
    }
    ty = typeof(|| false);
    o: ty = false;
    ";
    match full_parse(string).map(compile) {
        Ok(Ok(rules)) => println!("{rules:#?}"),
        Ok(Err(err)) => println!("{err}"),
        Err(err) => println!("{err}"),
    }
}

use crate::compiler::compile;
use crate::parser::full_parse;

mod tokenizer {
    use std::iter::Peekable;
    use std::mem::replace;
    use std::rc::Rc;
    use std::str::Chars;

    #[derive(Debug, Clone, Copy)]
    pub enum Keyword {
        Boolean(bool),
        FuncDef,
        Map,
        Foreach,
        If,
        Else,
        Input,
        Output,
    }

    fn get_keyword(string: &str) -> Option<Keyword> {
        match string {
            "true" => Some(Keyword::Boolean(true)),
            "false" => Some(Keyword::Boolean(false)),
            "comp" => Some(Keyword::FuncDef),
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
    }

    impl std::fmt::Display for Keyword {
        fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                Keyword::Boolean(val) => write!(formatter, "{val}"),
                Keyword::FuncDef => write!(formatter, "comp"),
                Keyword::Input => write!(formatter, "INPUT"),
                a => todo!("{a:?}"),
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
        Boolean(bool),
        Select(Box<Expression>, Box<Expression>, Box<Expression>),
        VarAccess(Rc<str>),
        TypedVarAccess(Rc<str>, Box<Expression>),
        Circular(Box<Expression>),
        Tuple(Vec<Expression>),
        Index(Box<Expression>, Index),
        Call(Box<Expression>, Vec<Expression>),
        Block(Vec<Rule>, Box<Expression>),
        Component(Vec<(Rc<str>, Expression)>, Box<Expression>, Box<Expression>),
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
            (Ok(Token::Ident(_)), Ok(Token::Colon | Token::Equals)) => Some(parse_rule(tokens)),
            (Ok(Token::Keyword(Keyword::FuncDef)), _) => Some(parse_func_def(tokens)),
            _ => None,
        }
    }

    fn parse_func_def(tokens: &mut TokenStream) -> Result<Rule, ParseError> {
        assert_token!(tokens, Token::Keyword(Keyword::FuncDef));
        let name = match_token!(tokens, Token::Ident(name) => name);
        assert_token!(tokens, Token::OpeningParen);
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
            Expression::Component(args, Box::new(ret), Box::new(block)),
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
                | E::VarAccess(_)
                | E::Component(..)
                | E::Block(..)
                | E::Boolean(_) => (),
                expr => todo!("{expr:?}"),
            }
        }
        Ok(Rule(name, reqs, expr))
    }

    fn parse_expression(tokens: &mut TokenStream) -> Result<Expression, ParseError> {
        let mut res = match_token!(tokens,
            Token::At => Expression::Circular(Box::new(parse_expression(tokens)?)),
            Token::Keyword(Keyword::Boolean(val)) => Expression::Boolean(val),
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
            Token::OpeningParen => match (tokens.peek()?, tokens.peek2()?) {
                (Token::Ident(_), Token::Colon) => parse_component(tokens),
                (Token::ClosingParen, Token::Arrow) => parse_component(tokens),
                _ => parse_tuple(tokens),
            }?,
            Token::OpeningBrace => parse_block(tokens)?,
            Token::Ident(name) => match tokens.peek() {
                Ok(Token::OpeningAngle) => {
                    tokens.token()?;
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
                        Token::Comma => (),
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
                Token::Ident(name) => {
                    assert_token!(tokens, Token::Colon);
                    args.push((name, parse_expression(tokens)?));
                },
                Token::ClosingParen => break,
            );
            match_token!(tokens,
                Token::Comma => (),
                Token::ClosingParen => break,
            );
        }
        assert_token!(tokens, Token::Arrow);
        Ok(Expression::Component(
            args,
            Box::new(parse_expression(tokens)?),
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
        U(usize),
    }
    #[derive(Debug, Clone, PartialEq)]
    pub enum Type {
        Primitive(Primitive),
        Tuple(Vec<Type>),
        Function(Vec<Type>, Box<Type>),
    }
    #[derive(Debug, Clone)]
    pub enum TypeError {
        UndefinedIdentifier(Rc<str>),
        CannotIndex(Type),
        CannotCall(Type),
        MismatchedTypes(Type, Type),
        OutOfBoundsIndexing(usize, usize),
        MissingArgument(Type),
        ExtraArgument(Type),
    }

    #[derive(Debug, Clone, PartialEq)]
    struct PartiallyCompiledRule(pub Rc<str>, pub PartiallyCompiledExpression);
    #[derive(Debug, Clone, PartialEq)]
    enum PartiallyCompiledExpression {
        Input(usize),
        Builtin(Primitive),
        Boolean(bool),
        Select(
            Box<PartiallyCompiledExpression>,
            Box<PartiallyCompiledExpression>,
            Box<PartiallyCompiledExpression>,
        ),
        // Circular(Type),
        Tuple(Vec<PartiallyCompiledExpression>),
        Component(Vec<(Rc<str>, Type)>, Type, Box<Expression>),
        Number(u128),
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
        use PartiallyCompiledExpression as PCE;
        match expr {
            PCE::Input(_) => Type::Primitive(Primitive::Bool),
            PCE::Boolean(_) => Type::Primitive(Primitive::Bool),
            PCE::Number(_) => Type::Primitive(Primitive::Numeral),
            PCE::Tuple(exprs) => Type::Tuple(exprs.iter().map(find_type).collect()),
            PCE::Component(args, ret, _) => Type::Function(
                args.iter().map(|x| x.1.clone()).collect(),
                Box::new(ret.clone()),
            ),
            PCE::Select(_, _, _) => Type::Primitive(Primitive::Bool),
            expr => todo!("{expr:#?}"),
        }
    }

    fn evaluate_type(expr: PartiallyCompiledExpression) -> Result<Type, TypeError> {
        use PartiallyCompiledExpression as PCE;
        Ok(match expr {
            PCE::Builtin(prim) => Type::Primitive(prim),
            PCE::Tuple(exprs) => Type::Tuple(
                exprs
                    .into_iter()
                    .map(evaluate_type)
                    .collect::<Result<_, _>>()?,
            ),
            expr => todo!("{expr:#?}"),
        })
    }

    fn reduce_expr(
        expr: Expression,
        context: &Context,
    ) -> Result<PartiallyCompiledExpression, TypeError> {
        use Expression as E;
        use PartiallyCompiledExpression as PCE;
        Ok(match expr {
            E::Boolean(val) => PCE::Boolean(val),
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
            E::Select(a, b, c) => PCE::Select(
                Box::new(reduce_expr(*a, context)?),
                Box::new(reduce_expr(*b, context)?),
                Box::new(reduce_expr(*c, context)?)
            ),
            // E::Circular(ty) => PCE::Circular(reduce_expr(*ty, context).and_then(evaluate_type)?),
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
                            return Err(TypeError::OutOfBoundsIndexing(index, exprs.len()));
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
            E::Component(args, ret, expr) => PCE::Component(
                args.into_iter()
                    .map(|(n, x)| {
                        reduce_expr(x, context)
                            .and_then(evaluate_type)
                            .map(|x| (n, x))
                    })
                    .collect::<Result<_, _>>()?,
                reduce_expr(*ret, context).and_then(evaluate_type)?,
                expr,
            ),
            expr => todo!("{expr:#?}\n{context:?}"),
        })
    }

    fn reduce_call(
        func: Expression,
        args: Vec<Expression>,
        context: &Context,
    ) -> Result<PartiallyCompiledExpression, TypeError> {
        let ((arg_names, mut arg_types), ret, expr): ((Vec<_>, Vec<_>), _, _) = match reduce_expr(func, context)? {
            PartiallyCompiledExpression::Component(a, b, c) => (a.into_iter().unzip(), b, *c),
            expr => return Err(TypeError::CannotCall(find_type(&expr))),
        };
        if arg_types.len() > args.len() {
            return Err(TypeError::MissingArgument(arg_types.swap_remove(args.len())));
        }
        let args = args
            .into_iter()
            .map(|x| reduce_expr(x, context))
            .collect::<Result<Vec<_>, _>>()?;
        if args.len() > arg_types.len() {
            return Err(TypeError::ExtraArgument(find_type(&args[arg_types.len()])));
        }
        if let Some((t1, t2)) = arg_types
            .into_iter()
            .zip(args.iter().map(find_type))
            .find(|(x, y)| x != y)
        {
            return Err(TypeError::MismatchedTypes(t1, t2));
        }
        let mut context = context.clone();
        arg_names
            .into_iter()
            .zip(args)
            .for_each(|(n, v)| add_to_context(n, v, &mut context));
        let expr = reduce_expr(expr, &context)?;
        let ret_actual = find_type(&expr);
        if ret_actual != ret {
            return Err(TypeError::MismatchedTypes(ret_actual, ret))
        }
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

    fn reduce_rules(
        rules: Vec<Rule>,
        context: &mut Context,
    ) -> Result<(), TypeError> {
        rules
            .into_iter()
            .map(|x| {
                let mut reqs =
                    x.1.into_iter()
                        .map(|x| reduce_expr(x, &*context).and_then(evaluate_type));
                let expr = reduce_expr(x.2, &*context)?;
                let expr_type = find_type(&expr);
                if let Some(t) = reqs.find(|x| x.as_ref().map_or(true, |x| x != &expr_type)) {
                    t.map(|x| Err(TypeError::MismatchedTypes(x, expr_type)))??;
                }
                add_to_context(x.0, expr, context);
                Ok(())
            })
            .collect()
    }

    fn reduce(
        rules: Vec<Rule>,
        inputs: Vec<Rc<str>>,
    ) -> Result<Context, TypeError> {
        use PartiallyCompiledExpression as PCE;
        let mut context = HashMap::with_capacity(inputs.len() + 3);
        inputs
            .into_iter()
            .enumerate()
            .for_each(|(i, x)| add_to_context(x, PCE::Input(i), &mut context));
        add_to_context("bool".into(), PCE::Builtin(Primitive::Bool), &mut context);
        add_to_context("type".into(), PCE::Builtin(Primitive::Type), &mut context);
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

    #[derive(Debug, Clone)]
    pub enum Value {
        Boolean(bool),
        Input(usize),
        Select(Box<Value>, Box<Value>, Box<Value>)
    }

    fn resolve_expr(expr: PartiallyCompiledExpression) -> Result<Value, TypeError> {
        use PartiallyCompiledExpression as PCE;
        use Value as V;
        Ok(match expr {
            PCE::Boolean(val) => V::Boolean(val),
            PCE::Input(val) => V::Input(val),
            PCE::Select(a, b, c) => match resolve_expr(*a)? {
                V::Boolean(val) => if val {
                    resolve_expr(*b)?
                } else {
                    resolve_expr(*c)?
                },
                a => V::Select(
                    Box::new(a),
                    Box::new(resolve_expr(*b)?),
                    Box::new(resolve_expr(*c)?)
                ),
            },
            expr => todo!("{expr:#?}"),
        })
    }

    fn resolve(
        mut rules: Context,
        outputs: Vec<Rc<str>>,
    ) -> Result<Vec<Value>, TypeError> {
        outputs.into_iter().map(|x| match rules.remove(&x).and_then(|mut x| x.pop()) {
            Some(expr) => resolve_expr(expr),
            None => Err(TypeError::UndefinedIdentifier(x)),
        }).collect()
    }

    pub fn compile((inputs, outputs, statements): ParseResult) -> Result<Vec<Value>, TypeError> {
        resolve(reduce(statements, inputs)?, outputs)
    }
}

fn main() {
    let string = "
    INPUT in1 in2;
    OUTPUT out;
    a = [in1, in2];
    b = {
        b = a;
        a = false;
        b
    };
    comp c(d: bool) -> (bool, bool) {
        (select(d, b[0], false), select(d, false, b[0]))
    }
    out: bool = c(true)[1];
    ";
    match full_parse(string).map(compile) {
        Ok(Ok(rules)) => println!("{rules:#?}"),
        Ok(Err(err)) => println!("{err}"),
        Err(err) => println!("{err}"),
    }
}

use crate::parser::full_parse;
use crate::type_parser::find_types;

trait CallOn: Sized {
    fn call_on<T>(self, f: impl FnMut(Self) -> T) -> T;
}
impl<T: Sized> CallOn for T {
    fn call_on<U>(self, mut f: impl FnMut(T) -> U) -> U {
        f(self)
    }
}
mod tokenizer {
    use std::iter::Peekable;
    use std::mem::replace;
    use std::rc::Rc;
    use std::str::Chars;
    
    #[derive(Debug, Clone, Copy)]
    pub enum Keyword {
        True,
        False,
        FuncDef,
        Map,
        Foreach,
        If,
        Else,
    }
    
    fn get_keyword(string: &str) -> Option<Keyword> {
        match string {
            "true" => Some(Keyword::True),
            "false" => Some(Keyword::False),
            "comp" => Some(Keyword::FuncDef),
            "map" => Some(Keyword::Map),
            "foreach" => Some(Keyword::Foreach),
            "if" => Some(Keyword::If),
            "else" => Some(Keyword::Else),
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
                Keyword::True => write!(formatter, "true"),
                Keyword::False => write!(formatter, "false"),
                a => todo!("{a:?}")
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
                        _ => Token::Colon
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
                                    }
                                    else {
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
    pub struct Rule(pub Rc<str>, pub Vec<Type>, pub Expression);
    #[derive(Debug, Clone, PartialEq)]
    pub enum Expression {
        VarAccess(Rc<str>),
        TypedVarAccess(Rc<str>, Type),
        Circular(Type),
        Tuple(Vec<Expression>),
        Index(Box<Expression>, Index),
        Call(Box<Expression>, Vec<Expression>),
        Block(Vec<Rule>, Box<Expression>),
        Component(Vec<(Rc<str>, Type)>, Type, Box<Expression>),
        Number(u128, u32),
    }
    #[derive(Debug, Clone, PartialEq)]
    pub enum Index {
        Number(usize),
        Range(usize, usize),
    }
    #[derive(Debug, Clone, PartialEq)]
    pub enum Type {
        Tuple(Vec<Type>),
        Function(Vec<Type>, Box<Type>),
        Name(Rc<str>),
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

    fn parse_rule(tokens: &mut TokenStream) -> Result<Rule, ParseError> {
        let name = match_token!(tokens,
            Token::Ident(name) => name,
        );
        let ty = match tokens.peek()? {
            Token::Colon => {
                tokens.token()?;
                Some(parse_type(tokens)?)
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
                E::Circular(ty) => reqs.push(ty),
                E::Index(expr, _) => subs.push(*expr),
                E::Number(..) | E::VarAccess(_) | E::Component(..) | E::Block(..) => (),
                expr => todo!("{expr:?}"),
            }
        }
        Ok(Rule(name, reqs, expr))
    }

    fn parse_expression(tokens: &mut TokenStream) -> Result<Expression, ParseError> {
        let mut res = match_token!(tokens,
            Token::At => Expression::Circular(parse_type(tokens)?),
            Token::Numeral(num) => {
                let size = match tokens.peek() {
                    Ok(Token::Ident(string)) => {
                        let mut last = string.chars();
                        let first = last.next().ok_or(TokenError::EoF)?;
                        if first != 'u' {
                            return Err(ParseError::InvalidNumeral(num));
                        }
                        tokens.token()?;
                        last.collect::<String>().parse().map_err(|_| ParseError::InvalidNumeral(num))?
                    }
                    _ => 8,
                };
                if num.checked_shr(size).unwrap_or(0) > 0 {
                    return Err(ParseError::InvalidNumeral(num));
                }
                Expression::Number(num, size)
            },
            Token::Char(ch) => Expression::Number(ch as u128, 8),
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
                    let ty = parse_type(tokens)?;
                    assert_token!(tokens, Token::ClosingAngle);
                    Expression::TypedVarAccess(name, ty)
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
        let rules = parse_rules(tokens)?;
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
                    args.push((name, parse_type(tokens)?));
                },
                Token::ClosingParen => break,
            );
            match_token!(tokens,
                Token::Comma => (),
                Token::ClosingParen => break,
            );
        }
        assert_token!(tokens, Token::Arrow);
        let out = parse_type(tokens)?;
        Ok(Expression::Component(
            args,
            out,
            Box::new(parse_expression(tokens)?),
        ))
    }

    fn parse_type(tokens: &mut TokenStream) -> Result<Type, ParseError> {
        let (paren, mut ty) = match_token!(tokens,
            Token::Ident(name) => (false, Type::Name(name)),
            Token::OpeningParen => (true, parse_type_tuple(tokens)?),
        );

        loop {
            ty = match tokens.peek() {
                Ok(Token::OpeningBracket) => {
                    tokens.token()?;
                    let num = match_token!(tokens,
                        Token::Numeral(num) => num,
                    );
                    assert_token!(tokens, Token::ClosingBracket);
                    let types = vec![ty; num as usize];
                    Type::Tuple(types)
                }
                _ => break,
            };
        }

        match tokens.peek() {
            Ok(Token::Arrow) => {
                tokens.token()?;
                let ret = parse_type(tokens)?;
                Ok(Type::Function(
                    match (ty, paren) {
                        (Type::Tuple(types), true) => types,
                        (ty, _) => vec![ty],
                    },
                    Box::new(ret),
                ))
            }
            _ => Ok(ty),
        }
    }

    fn parse_type_tuple(tokens: &mut TokenStream) -> Result<Type, ParseError> {
        let mut types = vec![];
        match tokens.peek() {
            Ok(Token::ClosingParen) => {
                tokens.token()?;
                return Ok(Type::Tuple(types));
            }
            _ => {
                let ty = parse_type(tokens)?;
                match_token!(tokens,
                    Token::Comma => types.push(ty),
                    Token::ClosingParen => return Ok(ty),
                )
            }
        }
        loop {
            match tokens.peek() {
                Ok(Token::ClosingParen) => {
                    tokens.token()?;
                    break;
                }
                _ => {
                    types.push(parse_type(tokens)?);
                    match_token!(tokens,
                        Token::Comma => (),
                        Token::ClosingParen => break,
                    )
                }
            }
        }
        Ok(Type::Tuple(types))
    }

    fn parse_rules(tokens: &mut TokenStream) -> Result<Vec<Rule>, ParseError> {
        let mut rules = vec![];
        while let Ok(Token::Equals | Token::Colon) = tokens.peek2() {
            rules.push(parse_rule(tokens)?);
        }
        Ok(rules)
    }

    pub fn full_parse(string: &str) -> Result<Vec<Rule>, ParseError> {
        let tokens = &mut TokenStream::new(string);
        let ret = parse_rules(tokens)?;
        match tokens.token() {
            Err(TokenError::EoF) => (),
            Ok(token) => Err(ParseError::UnexpectedToken(
                token,
                vec!["Token::Identifier"],
                line!(),
            ))?,
            Err(err) => Err(err)?,
        }
        Ok(ret)
    }
}
mod type_parser {
    use crate::parser::*;
    use crate::CallOn;
    use std::rc::Rc;

    pub enum TypeError {
        UndefinedIdentifier(Rc<str>),
        CannotIndex(Type),
        CannotCall(Type),
        MismatchedTypes(Type, Type),
        OutOfBoundsIndexing(usize, usize),
    }

    impl std::fmt::Display for TypeError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self{
                TypeError::UndefinedIdentifier(name) => 
                    write!(formatter, "Undefined Identifier {name}\nNote: {name} may not be defined later in the program than it is used"),
                TypeError::CannotIndex(ty) => 
                    write!(formatter, "Cannot index thing of type {ty}"),
                TypeError::CannotCall(ty) => 
                    write!(formatter, "Cannot call thing of type {ty}"),
                TypeError::MismatchedTypes(type1, type2) => 
                    write!(formatter, "Expected {type1}, found {type2}"),
                TypeError::OutOfBoundsIndexing(len, index) => 
                    write!(formatter, "Index out of bounds: the len is {len} but the index is {index}"),
            }
        }
    }

    impl std::fmt::Display for Type {
        fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                Type::Name(name) => write!(formatter, "{name}"),
                Type::Tuple(types) => {
                    write!(formatter, "(")?;
                    for ty in types.iter() {
                        write!(formatter, "{},", ty)?;
                    }
                    write!(formatter, ")")
                }
                Type::Function(arg, ret) => {
                    write!(formatter, "(")?;
                    for ty in arg.iter() {
                        write!(formatter, "{},", ty)?;
                    }
                    write!(formatter, ")")?;
                    write!(formatter, " -> {ret}")
                }
            }
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct TypedRule(pub Rc<str>, pub TypedExpr);
    #[derive(Debug, Clone, PartialEq)]
    pub struct TypedExpr(pub Type, pub TypedExpression);
    #[derive(Debug, Clone, PartialEq)]
    pub enum TypedExpression {
        VarAccess(Rc<str>),
        Circular,
        Tuple(Vec<TypedExpr>),
        Index(Box<TypedExpr>, Index),
        Call(Box<TypedExpr>, Vec<TypedExpr>),
        Block(Vec<TypedRule>, Box<TypedExpr>),
        Component(Vec<Rc<str>>, Box<TypedExpr>),
        Number(u128, u32),
    }
    #[derive(Debug, Clone)]
    struct TypeAssoc(Rc<str>, Type);

    fn find_types_rule(rule: Rule, context: &[TypeAssoc]) -> Result<TypedRule, TypeError> {
        let expr = find_types_expr(rule.2, context)?;
        if let Some(ty) = rule.1.into_iter().find(|x| x != &expr.0) {
            return Err(TypeError::MismatchedTypes(ty, expr.0));
        }
        Ok(TypedRule(rule.0, expr))
    }

    fn find_types_expr(expr: Expression, context: &[TypeAssoc]) -> Result<TypedExpr, TypeError> {
        use Expression as E;
        use TypedExpr as T;
        use TypedExpression as TE;
        Ok(match expr {
            E::Number(val, size) => T(
                Type::Tuple(vec![Type::Name("bool".into()); size as usize]),
                TE::Number(val, size),
            ),
            E::VarAccess(name) => T(
                find_types_var(name.clone(), context, |_| true)?,
                TE::VarAccess(name),
            ),
            E::Index(expr, index) => match find_types_expr(*expr, context)? {
                T(Type::Tuple(types), expr) => T(
                    match index {
                        Index::Number(num) => {
                            if num >= types.len() {
                                return Err(TypeError::OutOfBoundsIndexing(types.len(), num));
                            }
                            types[num].clone()
                        }
                        Index::Range(start, end) => {
                            let reverse = start < end;
                            let (min, max) = if reverse { (start, end) } else { (end, start) };
                            if max >= types.len() {
                                return Err(TypeError::OutOfBoundsIndexing(types.len(), max));
                            }
                            let types = types[min..=max].to_vec();
                            Type::Tuple(if reverse {
                                types
                            } else {
                                types.into_iter().rev().collect()
                            })
                        }
                    },
                    TE::Index(Box::new(T(Type::Tuple(types), expr)), index),
                ),
                T(ty, _) => return Err(TypeError::CannotIndex(ty)),
            },
            E::Tuple(exprs) => exprs
                .into_iter()
                .map(|x| find_types_expr(x, context))
                .collect::<Result<Vec<_>, _>>()?
                .call_on(|x| {
                    T(
                        Type::Tuple(x.iter().map(|x| x.0.clone()).collect()),
                        TE::Tuple(x),
                    )
                }),
            E::Circular(ty) => T(ty, TE::Circular),
            E::Call(func, args) => match find_types_expr(*func, context)? {
                ref expr @ T(Type::Function(ref arg, ref ret), _) => {
                    let args: Vec<_> = args
                        .into_iter()
                        .map(|x| find_types_expr(x, context))
                        .collect::<Result<_, _>>()?;
                    if let Some((ty1, ty2)) = args.iter().zip(arg).find(|x| &x.0 .0 != x.1) {
                        return Err(TypeError::MismatchedTypes(ty2.clone(), ty1.0.clone()));
                    }
                    T(*ret.clone(), TE::Call(Box::new(expr.clone()), args))
                }
                T(ty, _) => return Err(TypeError::CannotCall(ty)),
            },
            E::Block(rules, expr) => {
                let (rules, context) = find_type_rules(rules, context)?;
                let expr = find_types_expr(*expr, &context)?;
                T(expr.0.clone(), TE::Block(rules, Box::new(expr)))
            }
            E::Component(args, ret, expr) => {
                let mut context = context.to_vec();
                context.extend(args.iter().cloned().map(|x| TypeAssoc(x.0, x.1)));
                // println!("{context:?}");
                let expr = find_types_expr(*expr, &context)?;
                if ret != expr.0 {
                    return Err(TypeError::MismatchedTypes(expr.0, ret));
                }
                let args = args.into_iter().unzip();
                T(
                    Type::Function(args.1, Box::new(ret)),
                    TE::Component(args.0, Box::new(expr)),
                )
            }
            a => todo!("{a:?}"),
        })
    }

    fn find_types_var(
        name: Rc<str>,
        context: &[TypeAssoc],
        predicate: impl Fn(&Type) -> bool,
    ) -> Result<Type, TypeError> {
        for assoc in context.iter().rev() {
            if assoc.0 == name && predicate(&assoc.1) {
                return Ok(assoc.1.clone());
            }
        }
        Err(TypeError::UndefinedIdentifier(name))
    }

    fn find_type_rules(
        rules: Vec<Rule>,
        context: &[TypeAssoc],
    ) -> Result<(Vec<TypedRule>, Vec<TypeAssoc>), TypeError> {
        let mut context = context.to_vec();
        let mut typed_rules = vec![];
        for rule in rules {
            let type_rule = find_types_rule(rule, &context)?;
            context.push(TypeAssoc(type_rule.0.clone(), type_rule.1 .0.clone()));
            typed_rules.push(type_rule);
        }
        Ok((typed_rules, context))
    }

    pub fn find_types(rules: Vec<Rule>) -> Result<Vec<TypedRule>, TypeError> {
        let mut context = vec![];
        let bool_ty = Type::Name("bool".into());
        let type_ty = Type::Name("type".into());
        let select_ty = Type::Function(vec![bool_ty.clone(); 3], Box::new(bool_ty.clone()));
        context.push(TypeAssoc("select".into(), select_ty));
        context.push(TypeAssoc("bool".into(), type_ty.clone()));
        Ok(find_type_rules(rules, &context)?.0)
    }
}
/*
// mod compiler {
//     use crate::unify_types::TypedRule;
//     use std::collections::HashMap;
//     use std::rc::Rc;

//     pub struct FinalLine(pub usize, pub usize, pub usize);

//     pub fn compile(unified: Vec<TypedRule>) -> (Vec<FinalLine>, Vec<usize>) {
//         todo!()
//     }
// }

// fn print_final_lines(lines: &[FinalLine], outputs: &[usize]) {
//     println!("0. 0");
//     println!("1. 1");
//     println!("2. clock");
//     println!("3. input0");
//     println!("4. input1");
//     println!("5. input2");
//     println!("6. input3");
//     println!("7. input4");
//     println!("8. input5");
//     println!("9. input6");
//     println!("10. input7");
//     for (i, line) in lines.iter().enumerate() {
//         if let Some(index) = outputs.iter().position(|x| *x == i + 11) {
//             print!("output {index}: ");
//         }
//         println!("{}. select(#{}, #{}, #{})", i + 11, line.0, line.1, line.2);
//     }
// }

// enum CompilationError {
//     ParseError(ParseError),
//     TypeError(TypeError),
// }

// impl From<ParseError> for CompilationError {
//     fn from(err: ParseError) -> CompilationError {
//         CompilationError::ParseError(err)
//     }
// }

// impl From<TypeError> for CompilationError {
//     fn from(err: TypeError) -> CompilationError {
//         CompilationError::TypeError(err)
//     }
// }

// impl std::fmt::Display for CompilationError {
//     fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
//         match self {
//             CompilationError::ParseError(err) => write!(formatter, "{}", err),
//             CompilationError::TypeError(err) => write!(formatter, "{}", err),
//         }
//     }
// }

// fn full_compile(code: &str) -> Result<(Vec<FinalLine>, Vec<usize>), CompilationError> {
//     let parsed = crate::parser::full_parse(code)?;
//     let types = find_types(&parsed)?;
//     let unified = unify(parsed, types);
//     Ok(compile(unified))
// }

// fn execute(
//     code: &[FinalLine],
//     outputs: &[usize],
//     inputs: impl IntoIterator<Item = u8>,
//     ticks: u32,
// ) -> Vec<Vec<bool>> {
//     let mut total_values = vec![false; code.len() + 11];
//     total_values[1] = true;
//     inputs
//         .into_iter()
//         .map(|input| {
//             for i in 0..8 {
//                 total_values[i + 3] = (input >> i) & 1 != 0;
//             }
//             for _ in 0..100 {
//                 total_values[2] = !total_values[2];
//                 for _ in 0..ticks {
//                     let slice = &code
//                         .iter()
//                         .map(|line| {
//                             total_values[if total_values[line.0] { line.1 } else { line.2 }]
//                         })
//                         .collect::<Vec<bool>>();
//                     total_values[11..(code.len() + 11)].copy_from_slice(slice);
//                 }
//             }
//             outputs.iter().map(|x| total_values[*x]).collect()
//         })
//         .collect()
// }
*/

fn main() {
    let string = "
    a = [true, false];
    b = {
        a = 1;
        b = 2;
        b
    };
    c = (a: bool, b: bool) -> (bool, bool)
        (select(a, b, false), select(a, false, b));
    output = c(true, false);
    ";
    match full_parse(string).map(find_types) {
        Ok(Ok(rules)) => println!("{rules:#?}"),
        Ok(Err(err)) => println!("{err}"),
        Err(err) => println!("{err}"),
    }
}

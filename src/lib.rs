use std::fmt::{Display, Formatter, Result};
use num::{Float, NumCast};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let slr_model = get_setup_data();
        assert_eq!(slr_model.get_x_average(), 7.4545455);
        assert_eq!(slr_model.get_y_average(), 4090.9092);
    }

    #[test]
    fn test_line_of_best_fit() {
        let slr_model = get_setup_data();
        let (y_intercept, slope) = slr_model.get_line_of_best_fit();
        let expected_y_intercept = 7836.259;
        let expected_slope = -502.42493;
        assert_eq!(y_intercept, expected_y_intercept);
        assert_eq!(slope, expected_slope);
    }

    #[test]
    fn test_r_squared() {
        let slr_model = get_setup_data();
        let r_squared = slr_model.get_r_squared();
        assert_eq!(r_squared, 0.91207063);
    }
}

#[allow(dead_code)]
fn get_setup_data() -> SLRModel<f32> {
    let x_vals: Vec<f32> = vec![4.0, 4.0, 5.0, 5.0, 7.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let y_vals: Vec<f32> = vec![6300.0, 5800.0, 5700.0, 4500.0, 4500.0, 4200.0, 4100.0, 3100.0, 2100.0, 2500.0, 2200.0];
    SLRModel::new(y_vals, x_vals)
}

#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Point {
        Point { x, y }
    }
}

#[allow(dead_code)]
pub struct SLRModel<T: Float> {
    // data_points: Vec<Point>,
    x: Vec<T>, 
    y: Vec<T>,
    x_average: T,
    y_average: T,
    slope: T,
    y_intercept: T,
    sum_square_errors: T,
    r_squared: T,
}

impl<T: Float> SLRModel<T> {
    pub fn new(y: Vec<T>, x: Vec<T>) -> SLRModel<T>  {
        let x_average = SLRModel::set_x_average(&x);
        let y_average = SLRModel::set_y_average(&y);
        let (y_intercept, slope) = SLRModel::calculate_line_of_best_fit(&x, &y, &x_average, &y_average);
        let sum_square_errors = SLRModel::calculate_sum_of_squared_errors(&x, &y, &x_average, &y_average);
        let r_squared = SLRModel::calculate_r_squared(&x, &y, &x_average, &y_average);
        SLRModel {
            x,
            y,
            x_average,
            y_average,
            slope,
            y_intercept,
            sum_square_errors,
            r_squared
        }
    }

    fn set_x_average(x_vals: &[T]) -> T {
        let x_sum: T = x_vals.iter().fold(NumCast::from(0.0).unwrap(), |acc, x| acc + *x);
        x_sum / NumCast::from(x_vals.len()).unwrap()
    }

    pub fn get_x_average(&self) -> T {
        self.x_average
    }

    fn set_y_average(y_vals: &[T]) -> T {
        let y_sum: T = y_vals.iter().fold(NumCast::from(0.0).unwrap(), |acc, y| acc + *y);
        y_sum / NumCast::from(y_vals.len()).unwrap()
    }

    pub fn get_y_average(&self) -> T {
        self.y_average
    }

    fn calculate_line_of_best_fit(x_vals: &[T], y_vals: &[T], x_average: &T, y_average: &T) -> (T, T) {
        let slope: T = SLRModel::calculate_slope(x_vals, y_vals, x_average, y_average);
        let y_intercept: T = *y_average - slope * *x_average;
        (y_intercept, slope)
    }

    pub fn get_line_of_best_fit(&self) -> (T, T) {
        (self.y_intercept, self.slope)
    }

    fn calculate_sum_of_squared_errors(x_vals: &[T], y_vals: &[T], x_average: &T, y_average: &T) -> T {
        let (y_intercept, slope) = SLRModel::calculate_line_of_best_fit(x_vals, y_vals, x_average, y_average);
        let sum_square_errors = x_vals.iter().zip(y_vals).fold(NumCast::from(0.0).unwrap(), |acc, (x, y)| acc + (*y - (y_intercept + slope * *x)).powi(2));
        sum_square_errors
    }

    pub fn get_sum_of_square_errors(&self) -> T {
        self.sum_square_errors
    }

    fn calculate_slope(x_vals: &[T], y_vals: &[T], x_average: &T, y_average: &T) -> T {
        let sum_squares_xy = SLRModel::calculate_sum_squares_xy(x_vals, y_vals, x_average, y_average);
        let sum_squares_xx = SLRModel::calculate_sum_squares_xx(x_vals, x_average);
        sum_squares_xy / sum_squares_xx
    }
    
    fn calculate_sum_squares_xy(x_vals: &[T], y_vals: &[T], x_average: &T, y_average: &T) -> T {
        let sum = x_vals.iter().zip(y_vals).fold(NumCast::from(0.0).unwrap(), |acc, (x, y)| acc + (*x - *x_average) * (*y - *y_average));
        sum
    }
    
    fn calculate_sum_squares_xx(x_vals: &[T], x_average: &T) -> T {
        let sum = x_vals.iter().fold(NumCast::from(0.0).unwrap(), |acc, x| acc + (*x - *x_average).powi(2));
        sum
    }

    fn calculate_total_sum_of_squares(y_vals: &[T], y_average: &T) -> T {
        let sum = y_vals.iter().fold(NumCast::from(0.0).unwrap(), |acc, y| acc + (*y - *y_average).powi(2));
        sum
    }

    fn calculate_r_squared(x_vals: &[T], y_vals: &[T], x_average: &T, y_average: &T) -> T {
        let r: T = NumCast::from(1.0).unwrap();
        r - (SLRModel::calculate_sum_of_squared_errors(x_vals, y_vals, x_average, y_average) / SLRModel::calculate_total_sum_of_squares(y_vals, y_average))
    }

    pub fn get_r_squared(&self) -> T {
        self.r_squared
    }
}

impl<T: Float + std::fmt::Display> Display for SLRModel<T> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let (y_intercept, slope) = self.get_line_of_best_fit();
        write!(f, "Line of best fit: y = {} + {}x \nR-squared: {}", y_intercept, slope, self.get_r_squared())
    }
}

trait SummaryPrint: Display {
    fn summary(&self) {
        println!("SLR Model");
        println!("---------");
        println!("{}", self.to_string());
    }
}

impl<T: Float + std::fmt::Display> SummaryPrint for SLRModel<T> {}
